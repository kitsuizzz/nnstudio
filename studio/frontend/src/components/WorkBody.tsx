import { useEffect, useRef, type HTMLAttributes, type ReactNode } from "react";
import ReactMarkdown, { type Components, type Options } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { padMathAnnotations } from "../lib/mathpad";

interface WorkBodyProps {
  content: string;
  error?: boolean;
  /** While the reply is still streaming, newly arrived text fades in instead of popping. */
  streaming?: boolean;
}

const baseComponents: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
};

/** Minimal mdast shape of the math nodes the upgrade walk touches. */
interface MathNode {
  type: string;
  value?: string;
  position?: { start: { offset?: number }; end: { offset?: number } };
  data?: { hProperties?: { className?: string[] } };
  children?: MathNode[];
}

/**
 * remark-math parses single-line `$$…$$` as *inline* math and drops the fact
 * that it was double-dollar — and models write display equations that way all
 * the time. When a math node's source span is delimited by `$$` (the vfile
 * still carries the source), ask KaTeX for display style instead.
 */
function doubleDollarDisplay() {
  return (tree: unknown, file: unknown): void => {
    const src = typeof (file as { value?: unknown } | undefined)?.value === "string" ? (file as { value: string }).value : "";
    const walk = (node: MathNode | undefined) => {
      for (const child of node?.children ?? []) {
        const s = child.position?.start.offset;
        const e = child.position?.end.offset;
        if (
          child.type === "inlineMath" &&
          typeof child.value === "string" &&
          s != null &&
          e != null &&
          src.slice(s, s + 2) === "$$" &&
          src.slice(e - 2, e) === "$$"
        ) {
          child.data ??= {};
          child.data.hProperties = { className: ["language-math", "math-display"] };
        }
        walk(child);
      }
    };
    walk(tree as MathNode);
  };
}

const remarkPlugins: Options["remarkPlugins"] = [remarkGfm, remarkMath, padMathAnnotations, doubleDollarDisplay];
// LaTeX math ($inline$, $$display$$) typeset by KaTeX. A malformed formula
// renders in the app's danger color instead of breaking the reply (rehype-katex
// falls back to a .katex-error span on KaTeX's parse errors).
const rehypePlugins: Options["rehypePlugins"] = [
  [rehypeKatex, { strict: false, errorColor: "var(--danger)" }],
];

const FRESH = "piece-fresh";

// Tags whose rendered content can straddle the "new text" boundary, so their
// text runs can be split and the fresh tail faded in.
const FRESH_TAGS = [
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "ul",
  "ol",
  "li",
  "blockquote",
  "pre",
  "code",
  "em",
  "strong",
  "del",
  "a",
  "s",
  "ins",
  "mark",
  "table",
  "thead",
  "tbody",
  "tr",
  "th",
  "td",
] as const;

interface FreshState {
  /** Length of the content as of the previous render — the fresh region starts here. */
  prevLen: number;
  /** Source start-offsets of elements that have ever been (partly) fresh.
   *  Sticky on purpose: when a block finishes and the fresh frontier moves
   *  past it, the class is kept so a fade still in flight is never cut
   *  short by a re-render, and settled text can never pop back. */
  freshEls: Set<number>;
  /** For a text node, source start-offset -> the frozen split point from the
   *  update that first crossed it. Frozen so the split (and its span) is
   *  stable across re-renders and the animation never restarts. */
  splits: Map<number, number>;
}

interface HastNode {
  type?: string;
  tagName?: string;
  value?: string;
  children?: HastNode[];
  position?: { start: { offset: number }; end: { offset: number } };
}

function renderTag(tag: string, rest: HTMLAttributes<HTMLElement>): ReactNode {
  if (tag === "a") return <a {...rest} target="_blank" rel="noreferrer" />;
  const Tag = tag as "p";
  return <Tag {...rest} />;
}

/**
 * Markdown overrides that fade in the fresh tail of a streaming reply.
 * The stream appends to `content`, so the fresh region is the character range
 * [prevLen, content.length) in source offsets — and hast nodes carry their
 * source positions, so each element knows exactly where it sits.
 * Elements that are wholly fresh fade as a whole; elements the boundary
 * crosses keep their stable prefix and wrap only the fresh tail of each text
 * run in a fading <span>.
 */
function buildFreshComponents(getState: () => FreshState, getNewStart: () => number): Components {
  const map: Record<string, unknown> = {};
  for (const tag of FRESH_TAGS) {
    map[tag] = (props: { node: HastNode } & HTMLAttributes<HTMLElement>) => {
      const state = getState();
      const newStart = getNewStart();
      const { node, ...rest } = props;
      const pos: HastNode["position"] | undefined = node?.position;
      if (!pos?.start || !pos?.end) return renderTag(tag, rest);
      const s = pos.start.offset;
      const e = pos.end.offset;
      if (newStart >= 0 && e > newStart) state.freshEls.add(s);
      const fresh = state.freshEls.has(s);
      const className = fresh ? (rest.className ? `${rest.className} ${FRESH}` : FRESH) : rest.className;
      const kids = node.children;
      // hast-util-to-jsx-runtime passes a single child as a scalar; normalize
      // so child indices always align with node.children.
      const childProps = !Array.isArray(rest.children) && kids?.length === 1 ? [rest.children] : rest.children;
      if (fresh && Array.isArray(kids) && Array.isArray(childProps) && childProps.length === kids.length) {
        const out: ReactNode[] = [];
        for (let i = 0; i < kids.length; i++) {
          const child = kids[i];
          if (child.type === "text" && child.position?.start) {
            const cs = child.position.start.offset;
            const ce = child.position.end.offset;
            const text = String(child.value ?? "");
            let k = state.splits.get(cs);
            if (k == null && newStart >= 0 && s < newStart && e > newStart && cs < newStart && ce > newStart) {
              k = newStart - cs;
              state.splits.set(cs, k);
            }
            if (k == null) {
              out.push(text);
            } else {
              if (k > 0) out.push(text.slice(0, k));
              if (k < text.length) out.push(<span key={i} className={FRESH}>{text.slice(k)}</span>);
            }
          } else {
            out.push((childProps as ReactNode[])[i]);
          }
        }
        return renderTag(tag, { ...rest, className, children: out });
      }
      return renderTag(tag, { ...rest, ...(fresh ? { className } : {}) });
    };
  }
  return { ...baseComponents, ...map } as unknown as Components;
}

/** The work itself: the assistant's response, typeset as Markdown. */
export default function WorkBody({ content, error = false, streaming = false }: WorkBodyProps) {
  const stateRef = useRef<FreshState>({ prevLen: 0, freshEls: new Set(), splits: new Map() });
  const newStartRef = useRef(-1);

  // The markdown overrides must keep a *stable* element type per tag. React
  // decides whether to update or remount a node by its component type; if we
  // handed ReactMarkdown a freshly built object (new closure identities) on
  // every render, each <p>/<li>/… would look like a brand-new component and
  // be remounted — restarting every fade on every token. So the overrides are
  // built exactly once and read the live frontier (prevLen) through a ref.
  const componentsRef = useRef<Components | null>(null);
  if (componentsRef.current === null) {
    componentsRef.current = buildFreshComponents(
      () => stateRef.current,
      () => newStartRef.current,
    );
  }

  useEffect(() => {
    stateRef.current.prevLen = content.length;
  }, [content]);

  if (error) {
    return <div className="piece-error">{content}</div>;
  }

  // While streaming — and for the renders right after, so an in-flight fade
  // can finish instead of snapping to full opacity — the fresh-tail overrides
  // stay mounted. Their frontier (prevLen) only ever moves forward, so nothing
  // already settled can re-fade. A work that never streamed keeps the plain
  // renderer: reopening an old conversation must not replay the fade.
  const useFresh = streaming || stateRef.current.freshEls.size > 0;
  newStartRef.current = useFresh ? stateRef.current.prevLen : -1;

  return (
    <div className="piece-md">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={useFresh ? componentsRef.current : baseComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

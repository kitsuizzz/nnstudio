/**
 * The conversation wall, re-unified as a sandbox "logic" brain.
 *
 * The brain is self-contained (no imports, no DOM) and runs in a sandboxed
 * iframe. The host pushes a live serializable projection of the conversation
 * as `ctx`; the brain pairs the flat messages into turns and re-emits the
 * registry blocks (a wallhead, one work per exchange, a docked composer).
 * WallFrame renders those blocks through the shared component library and owns
 * the wall chrome (scroll, running head, pinned composer). Door and retry
 * clicks are relayed to the host as `ai.host` calls.
 *
 * The source is one flat string with no `${` and no backticks, so it can be
 * embedded verbatim in the iframe's <script> and shown in the Code pane.
 */
export const wallSource = `
var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

function pad2(n) { var s = String(n); return s.length < 2 ? "0" + s : s; }
function formatTime(ms) { var d = new Date(ms); return pad2(d.getHours()) + ":" + pad2(d.getMinutes()); }
function formatDate(ms) { var d = new Date(ms); return d.getDate() + " " + MONTHS[d.getMonth()] + " " + d.getFullYear(); }

function compact(n) { if (n < 1000) return String(n); return (n / 1000).toFixed(1).replace(/.0$/, "") + "k"; }
// "· 12.4k in × 46 out tok/s" — the measured speed of an exchange.
function speedSuffix(speeds) {
  if (!speeds || !speeds.prefill) return "";
  return " · " + compact(speeds.prefill) + " in × " + (speeds.decode != null ? compact(speeds.decode) : "—") + " out tok/s";
}

var LONG_CHARS = 600, LONG_LINES = 12;
function isLongTitle(t) {
  if (t.length > LONG_CHARS) return true;
  var lines = 1;
  for (var i = 0; i < t.length; i++) if (t.charCodeAt(i) === 10) lines++;
  return lines > LONG_LINES;
}

// Pair the flat message list into exchanges: a user message opens a turn, the
// assistant message that follows closes it.
function pairTurns(messages) {
  var turns = [];
  for (var i = 0; i < messages.length; i++) {
    var m = messages[i];
    if (m.role === "user") {
      turns.push({ no: turns.length + 1, prompt: m });
      continue;
    }
    var last = turns[turns.length - 1];
    if (last && !last.response) { last.response = m; }
    else { turns.push({ no: turns.length + 1, prompt: { role: "user", content: "" }, response: m }); }
  }
  return turns;
}

function findStudio(id) {
  for (var j = 0; j < ctx.studios.length; j++) if (ctx.studios[j].id === id) return ctx.studios[j];
  return undefined;
}

function findThread(id) {
  var list = ctx.threads || [];
  for (var j = 0; j < list.length; j++) if (String(list[j].id) === String(id)) return list[j];
  return undefined;
}

// What the model is doing right now, for the streaming reply's status line
// (main conversations; build threads have the inline activity window instead).
function streamingStatus(response, opts) {
  if (ctx.kind === "build" || !opts.streaming || !response) return "";
  var tools = response.tools || [];
  for (var i = tools.length - 1; i >= 0; i--) {
    var ev = tools[i];
    if (ev.status === "pending") return "Waiting for your approval…";
    if (ev.status === "allowed") {
      if (ev.name === "web_fetch") return "Reading " + siteOf(ev.arg) + "…";
      return "Searching for “" + clip(ev.arg, 60) + "”…";
    }
  }
  return response.content ? "" : "Thinking…";
}
function clip(text, max) {
  var t = String(text == null ? "" : text).replace(/\\s+/g, " ").trim();
  return t.length > max ? t.slice(0, max) : t;
}
function siteOf(url) {
  var m = String(url).match(/^[a-z][a-z0-9+.-]*:\\/\\/([^/?#]+)/i);
  return clip((m ? m[1] : String(url)).replace(/^www\\./i, ""), 40);
}

// One exchange: your message hung on the right as a wall label, the reply as
// the work on the left.
function workBlock(turn, opts) {
  var prompt = turn.prompt, response = turn.response;
  var meta = response && response.meta;
  var agentName = (meta && meta.agentName) || opts.current.agentName;
  var model = (meta && meta.model) || opts.current.model;
  var effort = (meta && meta.effort) || opts.current.effort;
  var usage = meta && meta.usage;
  var speeds = meta && meta.speeds;
  var images = (prompt.images && prompt.images.length) ? prompt.images : [];
  var title = (prompt.content || "").trim();
  var long = isLongTitle(title);
  var showStudy = !!response && (!!response.thinking || opts.thinkingActive);
  var piece = [];
  if (response) {
    var statusText = streamingStatus(response, opts);
    if (response.content) piece.push({ type: "text", content: response.content, error: !!response.error, streaming: !!opts.streaming });
    else if (opts.streaming && !statusText) piece.push({ type: "caret" });
    if (statusText) piece.push({ type: "status", text: statusText });
    // The inline activity window (build threads only): what the AI is doing
    // right now — reasoning + raw writing — settling into "Done". Main
    // conversations stay clean: just the reply text and a thread card.
    if (ctx.kind === "build") {
      var raw = response.raw || "";
      var thinking = response.buildThinking || response.thinking || "";
      var hasHidden = !!raw && raw !== response.content;
      var criticLive = !!(response.critic && (response.critic.status === "reviewing" || response.critic.status === "refining"));
      var review = response.review;
      var reviewLive = !!(review && (review.phase === "rendering" || review.phase === "reviewing" || review.phase === "fixing"));
      var toolPhase = "";
      if (response.tools && response.tools.length) {
        for (var ti = 0; ti < response.tools.length; ti++) {
          var toolEvent = response.tools[ti];
          if (toolEvent.status === "pending" || toolEvent.status === "allowed") {
            toolPhase = toolEvent.name === "web_fetch" ? "Fetching the page" : "Searching the web";
          }
        }
      }
      if (thinking || hasHidden || response.building || response.studioPending || reviewLive || criticLive || toolPhase) {
        var phase = "Done";
        var active = false;
        if (toolPhase) { phase = toolPhase; active = true; }
        else if (response.studioPending) { phase = "Writing the spec"; active = true; }
        else if (reviewLive && review.phase === "rendering") { phase = "Rendering the app"; active = true; }
        else if (reviewLive && review.phase === "reviewing") { phase = "Inspecting the screenshots"; active = true; }
        else if (reviewLive && review.phase === "fixing") { phase = "Polishing"; active = true; }
        else if (response.building && response.plan) { phase = "Building"; active = true; }
        else if (response.building) { phase = "Planning the build"; active = true; }
        else if (response.awaitingBuild) { phase = "Plan ready"; active = false; }
        else if (criticLive && response.critic.status === "reviewing") { phase = "Reviewing the design"; active = true; }
        else if (criticLive && response.critic.status === "refining") { phase = "Polishing"; active = true; }
        piece.push({ type: "activity", phase: phase, active: active, thinking: thinking, writing: raw || response.content || "" });
      }
    }
    // The parent conversation's thread card: the build lives elsewhere.
    if (response.threadId != null) {
      var thread = findThread(response.threadId);
      if (thread) {
        piece.push({ type: "thread", appId: String(thread.id), title: thread.title, status: thread.status, active: !!thread.active });
      }
    }
    if (opts.open && response.question) {
      piece.push({ type: "question", question: response.question.question, options: response.question.options, assumptions: response.question.assumptions });
    }
    var awaiting = !!(response.awaitingBuild && response.plan && response.plan.todos && response.plan.todos.length);
    var pending = response.building || response.studioPending;
    if (awaiting) {
      // The plan is ready; the build starts only when the user says so.
      piece.push({ type: "plan", todos: response.plan.todos, active: response.plan.active, done: response.plan.done, pct: response.plan.pct, confirm: true, thinking: response.buildThinking, thinkingActive: false });
    } else if (pending) {
      if (response.plan && response.plan.todos && response.plan.todos.length) {
        piece.push({ type: "plan", todos: response.plan.todos, active: response.plan.active, done: response.plan.done, pct: response.plan.pct, thinking: response.buildThinking, thinkingActive: !!(response.building || response.studioPending) });
      } else {
        piece.push({ type: "preparing", text: response.building ? "Planning the build…" : undefined });
      }
    } else {
      if (response.studioFailed) {
        piece.push({ type: "interrupted", kind: response.studioFailed, raw: (response.studioFailed === "truncated" || response.studioFailed === "planning") ? response.studioRaw : undefined, canRetry: opts.canRetry });
      }
      if (opts.studio) piece.push({ type: "door", title: opts.studio.title, appId: opts.studio.id });
      var critic = response.critic;
      if (critic && (critic.status === "reviewing" || critic.status === "refining" || critic.status === "refined")) {
        piece.push({ type: "critic", status: critic.status });
      }
    }
  } else {
    piece.push({ type: "none" });
  }
  var label = {
    no: turn.no,
    live: opts.streaming,
    kicker: "You",
    title: title,
    long: long,
    time: prompt.at != null ? formatTime(prompt.at) : undefined,
    images: images.length ? images : undefined,
    artist: agentName,
    medium: "· " + model + " · effort " + effort,
    dims: usage
      ? (usage.prompt.toLocaleString() + " in × " + usage.completion.toLocaleString() + " out tokens" + speedSuffix(speeds))
      : (opts.streaming ? "counting tokens…" + speedSuffix(opts.speeds) : "— × — tokens"),
    study: showStudy
      ? { content: (response && response.thinking) || "", active: opts.thinkingActive, ms: response && response.thinkingMs }
      : undefined
  };
  return { type: "work", label: label, piece: piece, streaming: opts.streaming };
}

// Live data, pushed by the host. Reassigned wholesale on each ctx push.
var ctx = {
  messages: [],
  createdAt: null,
  conversationId: null,
  title: "",
  current: { agentName: "", model: "", effort: "medium" },
  streaming: false,
  firstTokenReceived: false,
  building: false,
  speeds: null,
  studios: [],
  threads: [],
  kind: "chat"
};

function render() {
  var messages = ctx.messages || [];
  var turns = pairTurns(messages);
  var lastTurn = turns[turns.length - 1];
  var lastResponse = lastTurn && lastTurn.response;
  var thinkingActive = ctx.streaming && ctx.firstTokenReceived && !!lastResponse && lastResponse.content === "";
  var count = turns.length;
  var date = formatDate(ctx.createdAt != null ? ctx.createdAt : Date.now());
  var head = {
    type: "wallhead",
    title: ctx.title || "New chat",
    meta: [
      { text: ctx.current.agentName },
      { text: ctx.current.model, model: true },
      { text: date },
      { text: count + (count === 1 ? " exchange" : " exchanges") }
    ]
  };
  var works = [];
  for (var i = 0; i < turns.length; i++) {
    works.push(workBlock(turns[i], {
      streaming: ctx.streaming && i === turns.length - 1,
      thinkingActive: i === turns.length - 1 && thinkingActive,
      open: i === turns.length - 1,
      current: ctx.current,
      studio: (turns[i].response && turns[i].response.studioId) ? findStudio(turns[i].response.studioId) : undefined,
      canRetry: i === turns.length - 1,
      speeds: ctx.streaming && i === turns.length - 1 ? ctx.speeds : null
    }));
  }
  // Flat list; WallFrame splits it back into head / works / dock by type.
  ai.render({ blocks: [head].concat(works, [{ type: "composer" }]) });
}

ai.on("ctx", function (data) { ctx = data || ctx; render(); });
ai.on("click", function (arg, blockType) {
  if (blockType === "door") ai.host("open-studio", arg);
  else if (blockType === "thread") ai.host("open-thread", arg);
  else if (blockType === "interrupted" && arg === "retry") ai.host("retry");
  else if (blockType === "plan" && arg === "build") ai.host("build");
  else if (blockType === "question") {
    if (arg === "skip") ai.host("skip-questions");
    else if (arg) ai.host("option", arg);
  }
});
// Announce readiness; the host answers with the current ctx, which renders.
ai.ready();
`;

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * The app shell's last line of defense. React unmounts the entire tree on an
 * uncaught render/commit error — with no boundary that means a blank page
 * ("the whole UI is gone"). This keeps a readable card with a reload instead,
 * and puts the error in the console for a bug report.
 */
export default class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("aistudio: the app crashed", error, info.componentStack);
  }

  render(): ReactNode {
    if (!this.state.error) return this.props.children;
    return (
      <div className="app-crash" role="alert">
        <h1>Something went wrong</h1>
        <p>
          The app hit an unexpected error. Your conversations are safe — reloading
          usually fixes it.
        </p>
        <pre>{String(this.state.error.message || this.state.error)}</pre>
        <button type="button" className="btn" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
    );
  }
}

import { Component, type ReactNode } from "react";

interface Props {
  /** Report the crash to the workspace debug log (and the AI). */
  onError: (error: Error) => void;
  /** Leave the workspace from the crash fallback. */
  onBack: () => void;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches host-side render crashes — a malformed spec can throw inside a block
 * renderer — so the workspace shows a recoverable card instead of a blank
 * window, and the crash lands in the debug log for the AI to fix.
 */
export default class StudioErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    this.props.onError(error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="studio-crash" data-surface="app">
          <p className="studio-crash-title">This workspace hit an error.</p>
          <pre className="studio-crash-message">{this.state.error.message || String(this.state.error)}</pre>
          <div className="studio-crash-actions">
            <button type="button" className="btn" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
            <button type="button" className="btn ghost" onClick={this.props.onBack}>
              Exit
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

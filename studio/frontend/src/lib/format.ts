export function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
}

export function formatDuration(ms: number): string {
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)} s`;
  }
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes} m ${String(seconds).padStart(2, "0")} s`;
}

/** 14:32 */
export function formatTime(epochMs: number): string {
  const d = new Date(epochMs);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** 8 September 2026 */
export function formatDate(epochMs: number): string {
  const d = new Date(epochMs);
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}

/** Today / Yesterday / Mon / 8 Sep — for dense tables. */
export function formatRelativeDay(epochMs: number, now = Date.now()): string {
  const d = new Date(epochMs);
  const today = new Date(now);
  const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  const dayMs = 86_400_000;
  const diffDays = Math.floor((startOfToday - new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()) / dayMs);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getDay()];
  return `${d.getDate()} ${MONTHS[d.getMonth()].slice(0, 3)}`;
}

/** "1,204 × 880" with em-dashes for unknowns. */
export function formatDimensions(input?: number, output?: number): string {
  const fmt = (n?: number) => (n == null ? "—" : n.toLocaleString());
  return `${fmt(input)} × ${fmt(output)}`;
}

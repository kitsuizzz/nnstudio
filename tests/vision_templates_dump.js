// The vision templates, as the UI builds them, on stdout as JSON: ui_boot's
// own bootstrap (DOM stub + page script) runs with DUMP_TEMPLATES set, so the
// docs that land on the canvas are the docs the engine tests then run.
//
//   node tests/vision_templates_dump.js
//
const { spawnSync } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const out = path.join(os.tmpdir(), "nnstudio_templates_" + process.pid + ".json");
const r = spawnSync(process.execPath, [path.join(__dirname, "ui_boot.js")],
                    { env: Object.assign({}, process.env, { DUMP_TEMPLATES: out }),
                      encoding: "utf8", timeout: 120000 });
if (!fs.existsSync(out)) {
  console.error(r.stdout || "");
  console.error(r.stderr || "");
  process.exit(1);
}
process.stdout.write(fs.readFileSync(out, "utf8"));
fs.unlinkSync(out);

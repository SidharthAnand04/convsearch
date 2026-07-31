"use strict";

const fs = require("node:fs");
const path = require("node:path");

const { TEST_WORKSPACE } = require("./paths");

/** Remove the disposable workspace. Guarded so a mis-edit can never delete real data. */
module.exports = async () => {
  const resolved = path.resolve(TEST_WORKSPACE);
  if (!path.basename(resolved).startsWith(".tmp-")) return;
  fs.rmSync(resolved, { recursive: true, force: true });
};

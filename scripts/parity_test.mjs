import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { predict } from "../netlify/functions/lib/inference.mjs";

const parity = JSON.parse(
  readFileSync(new URL("./parity-data.json", import.meta.url), "utf8")
);

let checked = 0;
let maxDelta = 0;

function check(row, source) {
  const expectedModels = row.models ?? row.expected;
  for (const [modelId, expected] of Object.entries(expectedModels)) {
    const result = predict(modelId, row.features);
    const delta = Math.abs(result.fraudProbability - expected.fraudProbability);
    maxDelta = Math.max(maxDelta, delta);
    assert.equal(
      result.prediction,
      expected.prediction,
      `${source}: ${modelId} prediction mismatch for index ${row.index}`
    );
    assert.ok(
      delta < 1e-6,
      `${source}: ${modelId} probability mismatch for index ${row.index}: ${result.fraudProbability} vs ${expected.fraudProbability}`
    );
    checked += 1;
  }
}

for (const row of parity.rows) {
  check(row, "random row");
}
for (const sample of parity.samples) {
  check(sample, "preset sample");
}

console.log(`Parity check passed: ${checked} predictions, max probability delta ${maxDelta}`);

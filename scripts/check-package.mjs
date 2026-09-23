import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { posix } from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url)));
const files = new Set(pkg.files);
for (const entry of Object.values(pkg.exports)) {
    const file = entry.replace(/^\.\//, '');
    assert.ok(file === 'package.json' || files.has(file), `Unpacked export: ${file}`);
}
for (const file of files) {
    const url = new URL(`../${file}`, import.meta.url);
    const source = readFileSync(url, 'utf8');
    if (!file.endsWith('.js')) continue;
    execFileSync(process.execPath, ['--check', fileURLToPath(url)]);
    for (const match of source.matchAll(/(?:from\s*|import\s*\()\s*['"](\.[^'"]+)['"]/g)) {
        const dependency = posix.normalize(posix.join(posix.dirname(file), match[1]));
        assert.ok(files.has(dependency), `${file} imports unpacked ${dependency}`);
    }
}
console.log('Package exports, source syntax, and relative dependencies: OK');

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = fileURLToPath(new URL('../', import.meta.url));
const bookDir = join(projectRoot, '_book');
const chapters = [
  'index',
  'softmax-cross-entropy',
  'embedding-geometry',
  'optimization',
  'attention',
  'transformer-architecture',
  'pretraining-scaling',
  'inference-systems',
  'rl-foundations',
  'rlhf-alignment',
  'frontier-rl',
];
const embeddedNavigationPattern = /<link\b(?=[^>]*\brel="(?:prev|next)")(?=[^>]*\bhref="data:text\/html,)[^>]*>/g;

for (let index = 0; index < chapters.length; index += 1) {
  const filename = `${chapters[index]}.html`;
  const path = join(bookDir, filename);
  let html = readFileSync(path, 'utf8');
  const replacements = [];

  html = html.replace(embeddedNavigationPattern, (tag) => {
    const rel = tag.match(/\brel="(prev|next)"/)[1];
    const targetIndex = rel === 'prev' ? index - 1 : index + 1;
    if (!chapters[targetIndex]) {
      throw new Error(`${filename} has an embedded ${rel} link outside the book`);
    }
    const target = `${chapters[targetIndex]}.html`;
    replacements.push(`${rel} -> ${target}`);
    return tag.replace(/href="[^"]*"/, `href="./${target}"`);
  });

  if (html.includes('data:text/html')) {
    throw new Error(`${filename} still contains an embedded navigation document`);
  }
  if (replacements.length > 0) {
    writeFileSync(path, html);
    console.log(`${filename}: ${replacements.join(', ')}`);
  }
}

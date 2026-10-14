import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
  },
  webServer: {
    command: 'node tests/support/static-server.mjs',
    url: 'http://127.0.0.1:4173/llm-book/softmax-cross-entropy.html',
    reuseExistingServer: !process.env.CI,
  },
});

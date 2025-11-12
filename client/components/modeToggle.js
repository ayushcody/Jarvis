export default class ModeToggle {
  constructor({ root, getServerUrl, onModeChange, log }) {
    this.root = root;
    this.getServerUrl = getServerUrl;
    this.onModeChange = onModeChange;
    this.log = log || (() => {});
    this.mode = 'conversation';
    this.button = document.createElement('button');
    this.button.className = 'mode-toggle-button';
    this.button.addEventListener('click', () => this.toggle());
    this.statusSpan = document.createElement('span');
  }

  async init() {
    await this.refresh();
  }

  async refresh() {
    const base = this.getServerUrl();
    if (!base) {
      return;
    }
    try {
      const res = await fetch(`${base}/mode`);
      if (res.ok) {
        const data = await res.json();
        this.mode = data.mode || 'conversation';
        this.render();
        this.onModeChange?.(this.mode);
      }
    } catch (err) {
      this.log('mode-refresh-failed', { error: err.message });
    }
  }

  async toggle() {
    const next = this.mode === 'conversation' ? 'push-to-talk' : 'conversation';
    await this.setMode(next);
  }

  async setMode(mode) {
    const base = this.getServerUrl();
    if (!base) {
      return;
    }
    try {
      const res = await fetch(`${base}/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (res.ok) {
        const data = await res.json();
        this.mode = data.mode || mode;
        this.render();
        this.onModeChange?.(this.mode);
        this.log('mode-change', { mode: this.mode });
      }
    } catch (err) {
      this.log('mode-change-error', { error: err.message });
    }
  }

  render() {
    if (!this.root) return;
    this.root.innerHTML = '';
    this.statusSpan.textContent = `Mode: ${this.mode === 'conversation' ? 'Conversation' : 'Push to Talk'}`;
    this.button.textContent = this.mode === 'conversation' ? 'Switch to Push-to-Talk' : 'Switch to Conversation';
    this.root.append(this.statusSpan, this.button);
  }
}

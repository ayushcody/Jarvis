export default class AudioController {
  constructor({ root, onHoldStart, onHoldEnd }) {
    this.root = root;
    this.onHoldStart = onHoldStart;
    this.onHoldEnd = onHoldEnd;
    this.mode = 'conversation';
    this.holdButton = document.createElement('button');
    this.holdButton.className = 'hold-button';
    this.holdButton.textContent = 'Hold to Talk';
    this._bindEvents();
    this.render();
  }

  _bindEvents() {
    const start = (ev) => {
      ev.preventDefault();
      this.onHoldStart?.();
      this.holdButton.classList.add('active');
    };
    const end = (ev) => {
      ev.preventDefault();
      this.onHoldEnd?.();
      this.holdButton.classList.remove('active');
    };
    this.holdButton.addEventListener('mousedown', start);
    this.holdButton.addEventListener('touchstart', start, { passive: false });
    window.addEventListener('mouseup', end);
    window.addEventListener('touchend', end);
  }

  setMode(mode) {
    this.mode = mode;
    this.render();
  }

  render() {
    if (!this.root) return;
    this.root.innerHTML = '';
    if (this.mode === 'push-to-talk') {
      this.root.append(this.holdButton);
    }
  }
}

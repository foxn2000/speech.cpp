// The speak panel: a text and the synthesis model's options, the speech played as the server makes it, and below it the
// result, which plays again, pauses, moves, saves as a WAVE file and goes to the transcribe panel; and for a model that
// takes voice files, a voice made from a recording.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { scriptLanguage } from './languages.js';
import { OptionForm, withServerDefaults } from './options.js';
import { Player } from './player.js';
import { save } from './wav.js';
import { Waveform } from './waveform.js';

const $ = (id) => document.getElementById(id);

const STOPS = {
  max_seconds: 'stopped at max seconds',
  model_limit: 'stopped at the longest speech the model makes',
};

function clock(seconds) {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

export class SpeakPanel {
  #held = null;
  #options = null;
  #player = new Player();
  #controller = null;
  #frame = 0;
  #voice = new AudioInput($('voice-audio'));
  #waveform = new Waveform($('speech-wave'), $('speech-seek'), (seconds) => {
    this.#player.seek(seconds);
    this.#paint();
  });

  /** `transcribe(wav, label)` hands the speech to the transcribe panel. */
  constructor(transcribe) {
    $('speak-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#speak();
    });
    $('speak-text').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) $('speak-form').requestSubmit();
    });
    $('speak-text').addEventListener('input', () => this.#options?.suggestVoice(scriptLanguage($('speak-text').value)));
    $('speak-stop').addEventListener('click', () => {
      this.#controller?.abort();
      this.#player.pause();
    });
    $('speech-play').addEventListener('click', () => (this.#player.playing ? this.#player.pause() : this.#player.play()));
    $('speech-save').addEventListener('click', () => save(this.#player.wav(), 'speech.wav'));
    $('speech-transcribe').addEventListener('click', () => transcribe(this.#player.wav(), `Speech of ${this.#held.model.name}`));
    this.#player.onchange = () => this.#animate();
    $('voice-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#addVoice();
    });
  }

  /** Shows the synthesis model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#held = held;
    $('speak-form').hidden = held === null;
    $('voice-maker').hidden = !held?.model.voice_recordings;
    // A model that speaks only in voices made from recordings has nothing to speak with until one is made.
    if (held?.model.voice_recordings && held.model.voices.length === 0) $('voice-maker').open = true;
    if (!held) {
      $('speech-result').hidden = true;
      return;
    }
    const kept = this.#options?.values() ?? {};
    this.#options = new OptionForm(withServerDefaults(held), $('speak-options'), $('speak-more'));
    this.#options.restore(kept);
    this.#options.suggestVoice(scriptLanguage($('speak-text').value));
  }

  #status(text) {
    $('speak-status').textContent = text;
  }

  /** Shows a failure at the field of the input it names, or under the form. */
  #fail(error) {
    if (error.param === 'input') {
      $('speak-text').setAttribute('aria-invalid', 'true');
      $('speak-text-error').textContent = error.message;
      $('speak-text-error').hidden = false;
      $('speak-text').focus();
      return;
    }
    if (error.param && this.#options.showError(error.param, error.message)) return;
    $('speak-error').textContent = error.message;
    $('speak-error').hidden = false;
  }

  #clearErrors() {
    this.#options.clearErrors();
    $('speak-text').removeAttribute('aria-invalid');
    $('speak-text-error').hidden = true;
    $('speak-error').hidden = true;
  }

  /** Keeps the waveform and the time in step with the player while it plays. */
  #animate() {
    cancelAnimationFrame(this.#frame);
    this.#paint();
    if (this.#player.playing || this.#controller) this.#frame = requestAnimationFrame(() => this.#animate());
  }

  #paint() {
    const player = this.#player;
    this.#waveform.update(player.chunks, player.duration, player.position);
    $('speech-time').textContent = `${clock(player.position)} / ${clock(player.duration)}`;
    $('speech-play').setAttribute('aria-label', player.playing ? 'Pause' : 'Play');
    $('speech-play').classList.toggle('playing', player.playing);
  }

  /** Speaks the text, one synthesis at a time, however it is asked for: the button, or Ctrl+Enter while it is hidden. */
  async #speak() {
    if (this.#controller) return;
    const text = $('speak-text');
    this.#clearErrors();
    if (!text.value.trim()) {
      this.#fail({ param: 'input', message: 'Write the text to speak.' });
      return;
    }
    this.#controller = new AbortController();
    $('speak-start').hidden = true;
    $('speak-stop').hidden = false;
    $('speech-result').hidden = true;
    this.#status('Starting');
    const model = this.#held.model.name;
    const started = performance.now();
    let firstAudio = 0;
    let done = null;
    try {
      for await (const event of api.speak({ input: text.value, ...this.#options.values() }, this.#controller.signal)) {
        if (event.type === 'start') {
          await this.#player.begin(event.rate);
        } else if (event.type === 'audio') {
          if (!firstAudio) {
            firstAudio = performance.now();
            $('speech-result').hidden = false;
            this.#setResultEnabled(false);
          }
          this.#player.push(event.samples);
          this.#status('Speaking');
          this.#animate();
        } else {
          done = event;
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') this.#fail(error);
    } finally {
      this.#controller = null;
      $('speak-start').hidden = false;
      $('speak-stop').hidden = true;
      this.#status('');
    }
    if (!firstAudio) return;
    const parts = [model, `time to first audio ${((firstAudio - started) / 1000).toFixed(2)} s`,
      `made in ${((performance.now() - started) / 1000).toFixed(2)} s`];
    if (done) parts.push(`seed ${done.seed}`);
    if (done && STOPS[done.stop]) parts.push(STOPS[done.stop]);
    if (!done) parts.push('stopped before its end');
    $('speech-meta').textContent = parts.join(' · ');
    this.#setResultEnabled(true);
    this.#animate();
  }

  #setResultEnabled(enabled) {
    for (const id of ['speech-play', 'speech-seek', 'speech-save', 'speech-transcribe']) $(id).disabled = !enabled;
  }

  async #addVoice() {
    const name = $('voice-name');
    const error = $('voice-error');
    error.hidden = true;
    name.removeAttribute('aria-invalid');
    const fail = (message) => {
      error.textContent = message;
      error.hidden = false;
    };
    if (!this.#voice.given) return fail('Drop, choose or record a recording of the voice first.');
    if (!name.value.trim()) {
      name.setAttribute('aria-invalid', 'true');
      name.focus();
      return fail('Name the voice, so that you can choose it.');
    }
    $('voice-add').disabled = true;
    $('voice-status').textContent = 'Making the voice';
    try {
      const recording = await this.#voice.wav(this.#held.model.sample_rate);
      this.show(await api.addVoice(name.value.trim(), recording));
      this.#options.select('voice', name.value.trim());
      $('voice-status').textContent = `Added ${name.value.trim()}, chosen above.`;
      name.value = '';
    } catch (e) {
      $('voice-status').textContent = '';
      if (e.param === 'file') this.#voice.fail(e.message);
      else fail(e.message);
    } finally {
      $('voice-add').disabled = false;
    }
  }
}

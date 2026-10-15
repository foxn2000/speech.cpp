// The fields of a model's request options, built from its information as speech info --json gives it: each option's
// type, whether it is required, its default, its range and its choices. The common options are shown and the rest are
// folded away; an option the model checks but does not steer by has no field, and a field left as it came sends
// nothing, so that the default applies: the model's own, or the one the server was started with. A text field starts
// with its default, so that emptying it sends the empty text in place of a default the server was started with.

import { languageName } from './languages.js';

const COMMON = new Set(['voice', 'language', 'speed', 'seed', 'instructions', 'prompt']);
const LONG = new Set(['instructions', 'prompt']);
/** A bound at or past which a range says nothing a user needs: the largest int32 and 2^53 - 1 among them. */
const UNBOUNDED = 1e6;
let fieldCount = 0;

function label(name) {
  return name[0].toUpperCase() + name.slice(1).replaceAll('_', ' ');
}

/** The range of a number as a user reads it: "0 to 1", "1 or more", "above 0", or nothing for one without bounds. */
function range(option) {
  const low = option.minimum ?? option.exclusive_minimum;
  const high = option.maximum ?? option.exclusive_maximum;
  const hasLow = low !== undefined && Math.abs(low) < UNBOUNDED;
  const hasHigh = high !== undefined && Math.abs(high) < UNBOUNDED;
  const above = option.exclusive_minimum !== undefined;
  if (hasLow && hasHigh) return above ? `above ${low}, up to ${high}` : `${low} to ${high}`;
  if (hasLow) return above ? `above ${low}` : `${low} or more`;
  if (hasHigh) return `up to ${high}`;
  return '';
}

/**
 * A held model's information with the defaults the server was started with in place of the model's own, so that each
 * field shows, and leaves to the server, the default a request without it gets; the model's own stays as `own`, which
 * a choice of the model's own default that is not among its choices, such as a language's auto, still offers.
 */
export function withServerDefaults(held) {
  const defaults = held.defaults ?? {};
  const options = held.model.options.map((o) =>
    o.name in defaults ? { ...o, default: defaults[o.name], own: o.default, server: true } : o);
  return { ...held.model, options };
}

function hint(option) {
  if (option.name === 'seed' && !option.server) return 'drawn for each request when empty';
  const parts = [];
  const shown = option.default !== undefined && (option.type === 'int' || option.type === 'float');
  if (shown) parts.push(`default ${option.default}${option.server ? ', set when the server started' : ''}`);
  else if (option.server) parts.push('default set when the server started');
  if (option.type === 'int' || option.type === 'float') parts.push(range(option));
  return parts.filter(Boolean).join(', ');
}

export class OptionForm {
  #fields = new Map();
  #voiceChosen = false;

  /**
   * Builds the fields of `info`'s options into `visible`, and the uncommon ones into `more`, the folded part, which is
   * hidden when it holds none; `only`, when given, names the options to build, those the endpoint's form takes, which
   * all go into `visible`.
   */
  constructor(info, visible, more = null, only = null) {
    visible.replaceChildren();
    more?.replaceChildren();
    for (const option of info.options) {
      if (!option.steers || (only && !only.has(option.name))) continue;
      const field = this.#build(option, info);
      this.#fields.set(option.name, field);
      (more && !COMMON.has(option.name) ? more : visible).append(field.element);
    }
    if (more) more.closest('details').hidden = more.children.length === 0;
  }

  #build(option, info) {
    const id = `option-${option.name}-${++fieldCount}`;
    const element = document.createElement('div');
    element.className = 'field';
    const caption = document.createElement('label');
    caption.htmlFor = id;
    caption.textContent = label(option.name);
    const help = document.createElement('p');
    help.className = 'hint';
    help.id = `${id}-hint`;
    help.textContent = hint(option);
    const error = document.createElement('p');
    error.className = 'error';
    error.id = `${id}-error`;
    error.hidden = true;

    let input;
    if (option.type === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = option.default ?? false;
      element.classList.add('check');
    } else if (option.name === 'voice') {
      input = this.#voices(option, info, help);
    } else if (option.choices) {
      input = document.createElement('select');
      if (!option.required) input.append(new Option(option.default !== undefined ? `Default (${this.#choiceText(option, option.default)})` : 'Default', ''));
      if (option.server && option.own !== undefined && !option.choices.includes(option.own)) {
        input.append(new Option(this.#choiceText(option, option.own), option.own));
      }
      for (const choice of option.choices) input.append(new Option(this.#choiceText(option, choice), choice));
    } else if (option.type === 'int' || option.type === 'float') {
      input = document.createElement('input');
      input.type = 'number';
      input.inputMode = option.type === 'int' ? 'numeric' : 'decimal';
      input.step = option.type === 'int' ? '1' : 'any';
      if (option.minimum !== undefined) input.min = option.minimum;
      if (option.maximum !== undefined && option.maximum < UNBOUNDED) input.max = option.maximum;
    } else {
      input = document.createElement(LONG.has(option.name) ? 'textarea' : 'input');
      if (LONG.has(option.name)) {
        input.rows = 2;
        element.classList.add('wide');
      }
      input.value = option.default ?? '';
    }
    input.id = id;
    input.name = option.name;
    input.setAttribute('aria-describedby', `${help.id} ${error.id}`);
    if (option.type === 'bool') element.append(input, caption, error);
    else element.append(caption, input, help, error);
    return { option, input, error, element };
  }

  #choiceText(option, choice) {
    return option.name === 'language' && choice !== 'auto' ? languageName(choice) : choice;
  }

  /** The voices, grouped by their language, with the chosen one's description below them. */
  #voices(option, info, help) {
    const select = document.createElement('select');
    const voices = new Map((info.voices ?? []).map((v) => [v.name, v]));
    if (option.choices.length === 0) {
      select.append(new Option('No voice yet', ''));
      select.disabled = true;
    }
    const groups = new Map();
    for (const choice of option.choices) {
      const language = voices.get(choice)?.language ?? '';
      if (!groups.has(language)) groups.set(language, []);
      groups.get(language).push(choice);
    }
    const ordered = [...groups.keys()].sort((a, b) => (a && b ? languageName(a).localeCompare(languageName(b), 'en') : a ? 1 : -1));
    for (const language of ordered) {
      const parent = language ? Object.assign(document.createElement('optgroup'), { label: languageName(language) }) : select;
      for (const choice of groups.get(language)) parent.append(new Option(choice, choice));
      if (parent !== select) select.append(parent);
    }
    // The description says what the name does not: the voice's character, and its gender where the model gives one.
    const describe = () => {
      const voice = voices.get(select.value);
      const none = option.choices.length === 0 && info.voice_recordings ? 'Make one from a recording below.' : '';
      help.textContent = voice ? voice.description || voice.gender : none;
    };
    select.addEventListener('change', (event) => {
      describe();
      if (event.isTrusted) this.#voiceChosen = true;
    });
    if (option.choices.includes(option.default)) select.value = option.default;
    // A voice the server was started with is kept as one the user chose, which the text's language does not replace.
    this.#voiceChosen = option.server === true;
    describe();
    this.voices = voices;
    return select;
  }

  /**
   * Chooses a voice of `language` for the text, unless the user chose one or the voice chosen is of that language
   * already.
   */
  suggestVoice(language) {
    const field = this.#fields.get('voice');
    if (!field || !language || this.#voiceChosen) return;
    if (this.voices.get(field.input.value)?.language === language) return;
    const match = [...this.voices.values()].find((v) => v.language === language && field.option.choices.includes(v.name));
    if (match) this.select('voice', match.name);
  }

  /** The values set, by option name and of the option's type, without the fields left as they came. */
  values() {
    const out = {};
    for (const [name, { option, input }] of this.#fields) {
      if (option.type === 'bool') {
        if (input.checked !== (option.default ?? false)) out[name] = input.checked;
      } else if (option.type === 'string' && !option.choices) {
        const value = input.value.trim() === '' ? '' : input.value;
        if (value !== (option.default ?? '')) out[name] = value;
      } else if (input.value.trim() !== '') {
        out[name] = option.type === 'int' || option.type === 'float' ? Number(input.value) : input.value;
      }
    }
    return out;
  }

  /** Sets the fields to `values`, as values() gave them, where the option and the choice are still there. */
  restore(values) {
    for (const [name, value] of Object.entries(values)) {
      const field = this.#fields.get(name);
      if (!field) continue;
      if (field.option.type === 'bool') field.input.checked = value;
      else if (!field.option.choices || field.option.choices.includes(value)) field.input.value = value;
      field.input.dispatchEvent(new Event('change'));
    }
  }

  select(name, value) {
    this.restore({ [name]: value });
  }

  /** Shows a failure at the field of the option it names, and returns whether there is one. */
  showError(name, message) {
    const field = this.#fields.get(name);
    if (!field) return false;
    field.error.textContent = message;
    field.error.hidden = false;
    field.input.setAttribute('aria-invalid', 'true');
    field.element.closest('details')?.setAttribute('open', '');
    field.input.focus();
    return true;
  }

  clearErrors() {
    for (const { input, error } of this.#fields.values()) {
      error.hidden = true;
      input.removeAttribute('aria-invalid');
    }
  }
}

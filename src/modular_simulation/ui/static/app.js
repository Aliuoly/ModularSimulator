const state = {
  metadata: null,
  sensors: [],
  controllers: [],
  calculations: [],
  plots: null,
};

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const contentType = response.headers.get('Content-Type') || '';
    let message = `Request failed with status ${response.status}`;
    if (contentType.includes('application/json')) {
      try {
        const payload = await response.json();
        if (payload && payload.error) {
          message = payload.error;
        }
      } catch (error) {
        // ignore JSON parsing errors and fall back to default message
      }
    } else {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get('Content-Type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

function populateSelect(select, options, preferred) {
  if (!select) return;
  const currentValue = select.value;
  const values = [];
  const addValue = (value) => {
    if (typeof value !== 'string') return;
    const normalized = value.trim();
    if (!normalized) return;
    if (!values.includes(normalized)) {
      values.push(normalized);
    }
  };

  (options || []).forEach((value) => addValue(value));
  if (typeof preferred === 'string') {
    addValue(preferred);
  }

  select.innerHTML = '';
  if (values.length === 0) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'Select unit';
    select.appendChild(option);
    select.value = '';
    select.disabled = true;
    return;
  }

  select.disabled = false;
  values.forEach((value) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });

  const desired = (() => {
    if (typeof preferred === 'string') {
      const normalized = preferred.trim();
      if (normalized && values.includes(normalized)) {
        return normalized;
      }
    }
    if (currentValue && values.includes(currentValue)) {
      return currentValue;
    }
    return values[0];
  })();

  select.value = desired;
}

function normalizeUnitMetadata(info) {
  if (!info || typeof info !== 'object') {
    return { unit: '', aliases: [] };
  }
  const unit = typeof info.unit === 'string' ? info.unit : '';
  const aliases = Array.isArray(info.unit_aliases)
    ? info.unit_aliases.filter((alias) => typeof alias === 'string' && alias.trim() !== '')
    : [];
  return { unit, aliases };
}

function createUnitHint(role = 'unit-hint') {
  const hint = document.createElement('div');
  hint.className = 'unit-hint';
  hint.dataset.role = role;

  const label = document.createElement('span');
  label.className = 'unit-hint__label';
  label.textContent = 'Reference unit';
  hint.appendChild(label);

  const unitText = document.createElement('code');
  unitText.dataset.role = 'unit-hint-text';
  unitText.textContent = '—';
  hint.appendChild(unitText);

  const aliasIcon = document.createElement('span');
  aliasIcon.className = 'info-icon';
  aliasIcon.dataset.role = 'unit-hint-alias';
  aliasIcon.textContent = 'i';
  aliasIcon.title = 'Aliases: —';
  hint.appendChild(aliasIcon);

  return hint;
}

function updateUnitHintElement(hintElement, unit, aliases = []) {
  if (!hintElement) return;
  const unitText = hintElement.querySelector('[data-role="unit-hint-text"]');
  const aliasIcon = hintElement.querySelector('[data-role="unit-hint-alias"]');
  const normalizedUnit = typeof unit === 'string' && unit.trim() !== '' ? unit : '—';
  if (unitText) {
    unitText.textContent = normalizedUnit;
  }
  if (aliasIcon) {
    const aliasList = Array.isArray(aliases) ? aliases.filter((alias) => alias && alias.trim()) : [];
    if (aliasList.length > 0) {
      aliasIcon.title = `Aliases: ${aliasList.join(', ')}`;
      aliasIcon.classList.remove('hidden');
    } else {
      aliasIcon.title = 'No aliases for this unit';
      aliasIcon.classList.remove('hidden');
    }
  }
}

function applyUnitMetadataToInput(input, hint, metadata, preferred, forceBase = false) {
  const unitValue = typeof metadata.unit === 'string' ? metadata.unit : '';
  const preferredValue = typeof preferred === 'string' ? preferred.trim() : '';
  if (input) {
    if (forceBase) {
      input.value = unitValue;
    } else if (preferredValue) {
      input.value = preferredValue;
    } else if (!input.value && unitValue) {
      input.value = unitValue;
    }
    input.placeholder = unitValue;
  }
  updateUnitHintElement(hint, unitValue, metadata.aliases || []);
}

function getMeasurableInfo(tag) {
  if (!state.metadata || !Array.isArray(state.metadata.measurables)) return null;
  return state.metadata.measurables.find((item) => item.tag === tag) || null;
}

function getControlElementInfo(tag) {
  if (!state.metadata || !state.metadata.control_element_units) return null;
  return state.metadata.control_element_units[tag] || null;
}

function getUsableTagInfo(tag) {
  if (!state.metadata || !state.metadata.usable_tag_units) return null;
  return state.metadata.usable_tag_units[tag] || null;
}

function readQuantityRangeUnits(value) {
  if (!value) return { lower: '', upper: '' };
  if (Array.isArray(value)) {
    const lower = value[0] && typeof value[0] === 'object' ? value[0].unit || '' : '';
    const upper = value[1] && typeof value[1] === 'object' ? value[1].unit || '' : '';
    return { lower, upper };
  }
  if (typeof value === 'object') {
    const lower = value.lower && typeof value.lower === 'object' ? value.lower.unit || '' : '';
    const upper = value.upper && typeof value.upper === 'object' ? value.upper.unit || '' : '';
    return { lower, upper };
  }
  return { lower: '', upper: '' };
}

function setupSensorUnitOptions(container, defaults = {}) {
  const measurementLabel = container.querySelector('label[data-field="measurement_tag"]');
  const unitLabel = container.querySelector('label[data-field="unit"]');
  if (!measurementLabel || !unitLabel) return;
  const measurementSelect = measurementLabel.querySelector('select');
  const measurementInput = measurementLabel.querySelector('input');
  const unitInput = unitLabel.querySelector('input[data-role="unit"]');
  const unitHint = unitLabel.querySelector('[data-role="unit-hint"]');
  if (!unitInput) return;

  const updateUnits = (preferred) => {
    let tag = '';
    if (measurementSelect) {
      tag = measurementSelect.value;
    } else if (measurementInput) {
      tag = measurementInput.value;
    } else if (typeof defaults.measurement_tag === 'string') {
      tag = defaults.measurement_tag;
    }
    const info = getMeasurableInfo(tag);
    const metadata = normalizeUnitMetadata(info);
    const forceBase = preferred === null;
    applyUnitMetadataToInput(unitInput, unitHint, metadata, preferred, forceBase);
  };

  updateUnits(typeof defaults.unit === 'string' ? defaults.unit : '');

  if (measurementSelect) {
    measurementSelect.addEventListener('change', () => updateUnits(null));
  } else if (measurementInput) {
    measurementInput.addEventListener('blur', () => updateUnits(null));
  }
}

function setupMvRangeUnits(container, defaults = {}) {
  const rangeLabel = container.querySelector('label[data-field="mv_range"]');
  if (!rangeLabel) return () => {};
  const lowerInput = rangeLabel.querySelector('input[data-role="lower-unit"]');
  const upperInput = rangeLabel.querySelector('input[data-role="upper-unit"]');
  const lowerHint = rangeLabel.querySelector('[data-role="lower-unit-hint"]');
  const upperHint = rangeLabel.querySelector('[data-role="upper-unit-hint"]');
  if (!lowerInput || !upperInput) return () => {};
  const mvTagLabel = container.querySelector('label[data-field="mv_tag"]');
  const mvTagSelect = mvTagLabel ? mvTagLabel.querySelector('select') : null;
  const mvTagInput = mvTagLabel ? mvTagLabel.querySelector('input') : null;
  const rangeDefaults = readQuantityRangeUnits(defaults.mv_range);

  const applyUnits = (preferredLower, preferredUpper) => {
    let tag = '';
    if (mvTagSelect) {
      tag = mvTagSelect.value;
    } else if (mvTagInput) {
      tag = mvTagInput.value;
    } else if (typeof defaults.mv_tag === 'string') {
      tag = defaults.mv_tag;
    }
    const info = getControlElementInfo(tag);
    const metadata = normalizeUnitMetadata(info);
    const lowerForce = preferredLower === null;
    const upperForce = preferredUpper === null;
    applyUnitMetadataToInput(lowerInput, lowerHint, metadata, preferredLower, lowerForce);
    applyUnitMetadataToInput(upperInput, upperHint, metadata, preferredUpper, upperForce);
  };

  applyUnits(rangeDefaults.lower, rangeDefaults.upper);

  if (mvTagSelect) {
    mvTagSelect.addEventListener('change', () => applyUnits(null, null));
  } else if (mvTagInput) {
    mvTagInput.addEventListener('blur', () => applyUnits(null, null));
  }

  return applyUnits;
}

function showError(message) {
  const panel = document.getElementById('error-banner');
  if (!panel) {
    alert(message);
    return;
  }
  const safeMessage = message || 'An unexpected configuration error occurred.';
  panel.innerHTML = `
    <div class="error-message">Configuration error: ${safeMessage}</div>
    <button type="button" class="error-dismiss">Dismiss</button>
  `;
  panel.classList.remove('hidden');
  const dismiss = panel.querySelector('.error-dismiss');
  if (dismiss) {
    dismiss.addEventListener('click', () => {
      clearError();
    });
  }
}

function clearError() {
  const panel = document.getElementById('error-banner');
  if (panel) {
    panel.classList.add('hidden');
    panel.innerHTML = '';
  }
}

function handleError(error) {
  console.error(error);
  const message = error && error.message ? error.message : 'An unexpected error occurred.';
  showError(message);
}

function renderMessages(messages) {
  const panel = document.getElementById('messages');
  if (!messages || messages.length === 0) {
    panel.classList.add('hidden');
    panel.innerHTML = '';
    return;
  }
  panel.classList.remove('hidden');
  panel.innerHTML = `<h2>System Messages</h2><ul>${messages.map((msg) => `<li>${msg}</li>`).join('')}</ul>`;
}

function renderMeasurables() {
  const body = document.querySelector('#measurables-table tbody');
  body.innerHTML = '';
  if (!state.metadata) return;
  state.metadata.measurables.forEach((item) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><code>${item.tag}</code></td>
      <td>${item.category.replace('_', ' ')}</td>
      <td>${item.unit}</td>
    `;
    body.appendChild(row);
  });
}

function renderSensorList() {
  const list = document.getElementById('sensor-list');
  list.innerHTML = '';
  state.sensors.forEach((sensor) => {
    const li = document.createElement('li');
    li.className = 'item-card';
    const params = sensor.params || {};
    const alias = params.alias_tag || params.measurement_tag;
    li.innerHTML = `
      <header>
        <h3>${sensor.type}</h3>
      </header>
      <div>Measurement: <span class="tag-pill">${params.measurement_tag}</span></div>
      <div>Alias: <span class="tag-pill">${alias}</span></div>
      <div class="actions">
        <button data-action="add-controller" data-sensor="${sensor.id}">Add Controller</button>
        <button data-action="remove" data-id="${sensor.id}">Remove</button>
      </div>
    `;
    list.appendChild(li);
  });

  list.querySelectorAll('button[data-action="remove"]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const id = event.currentTarget.dataset.id;
      try {
        await fetchJSON(`/api/sensors/${id}`, { method: 'DELETE' });
        await refreshSensors();
        await refreshMetadata();
        clearError();
      } catch (error) {
        handleError(error);
      }
    });
  });

  list.querySelectorAll('button[data-action="add-controller"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const sensor = state.sensors.find((s) => s.id === event.currentTarget.dataset.sensor);
      if (!sensor) return;
      const defaults = {
        cv_tag: sensor.params.alias_tag || sensor.params.measurement_tag,
      };
      openControllerForm({ defaults, parentId: null });
    });
  });
}

function renderControllerList() {
  const list = document.getElementById('controller-list');
  list.innerHTML = '';
  state.controllers.forEach((controller) => {
    const li = document.createElement('li');
    li.className = 'item-card';
    const params = controller.params || {};
    const trajectory = controller.trajectory || { segments: [] };
    li.innerHTML = `
      <header>
        <h3>${controller.type}</h3>
      </header>
      <div>CV: <span class="tag-pill">${params.cv_tag}</span></div>
      <div>MV: <span class="tag-pill">${params.mv_tag}</span></div>
      <div>Mode: ${params.mode ?? 'AUTO'}</div>
      <div>Segments: ${trajectory.segments.length}</div>
      <div class="actions">
        <button data-action="edit-controller" data-id="${controller.id}">Edit Settings</button>
        <button data-action="edit-trajectory" data-id="${controller.id}">Edit Trajectory</button>
        <button data-action="cascade" data-id="${controller.id}">Cascade To…</button>
        <button data-action="remove" data-id="${controller.id}">Remove</button>
      </div>
    `;
    list.appendChild(li);
  });

  list.querySelectorAll('button[data-action="remove"]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const id = event.currentTarget.dataset.id;
      try {
        await fetchJSON(`/api/controllers/${id}`, { method: 'DELETE' });
        await refreshControllers();
        await refreshMetadata();
        clearError();
      } catch (error) {
        handleError(error);
      }
    });
  });

  list.querySelectorAll('button[data-action="edit-trajectory"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const controller = state.controllers.find((c) => c.id === event.currentTarget.dataset.id);
      if (!controller) return;
      openTrajectoryEditor(controller);
    });
  });

  list.querySelectorAll('button[data-action="edit-controller"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const controller = state.controllers.find((c) => c.id === event.currentTarget.dataset.id);
      if (!controller) return;
      openControllerEditor(controller);
    });
  });

  list.querySelectorAll('button[data-action="cascade"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const controller = state.controllers.find((c) => c.id === event.currentTarget.dataset.id);
      if (!controller) return;
      const defaults = { cv_tag: controller.params.cv_tag, mv_tag: controller.params.mv_tag };
      openControllerForm({ defaults, parentId: controller.id });
    });
  });
}

function renderCalculationList() {
  const list = document.getElementById('calculation-list');
  list.innerHTML = '';
  state.calculations.forEach((calc) => {
    const li = document.createElement('li');
    li.className = 'item-card';
    const outputs = calc.outputs || [];
    li.innerHTML = `
      <header><h3>${calc.type}</h3></header>
      <div>Outputs: ${outputs.map((tag) => `<span class="tag-pill">${tag}</span>`).join(' ') || '—'}</div>
      <div class="actions">
        <button data-action="remove" data-id="${calc.id}">Remove</button>
      </div>
    `;
    list.appendChild(li);
  });

  list.querySelectorAll('button[data-action="remove"]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const id = event.currentTarget.dataset.id;
      try {
        await fetchJSON(`/api/calculations/${id}`, { method: 'DELETE' });
        await refreshCalculations();
        await refreshMetadata();
        clearError();
      } catch (error) {
        handleError(error);
      }
    });
  });
}

function renderPlotLayout() {
  if (!state.plots) return;
  const form = document.getElementById('plot-form');
  form.rows.value = state.plots.rows;
  form.cols.value = state.plots.cols;
  const container = document.getElementById('plot-lines');
  container.innerHTML = '';
  state.plots.lines.forEach((line, index) => {
    const card = document.createElement('div');
    card.className = 'item-card';
    card.innerHTML = `
      <header><h3>Panel ${line.panel + 1}</h3></header>
      <div>Tag: <span class="tag-pill">${line.tag}</span></div>
      <div>Label: ${line.label || '—'} | Color: ${line.color || 'auto'} | Style: ${line.style || 'solid'}</div>
      <div class="actions">
        <button data-action="edit" data-index="${index}" type="button">Edit</button>
        <button data-action="remove" data-index="${index}" type="button">Remove</button>
      </div>
    `;
    container.appendChild(card);
  });
  container.querySelectorAll('button[data-action="remove"]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      const idx = Number(event.currentTarget.dataset.index);
      const lines = state.plots.lines.filter((_, i) => i !== idx);
      try {
        await updatePlotLayout({ ...state.plots, lines });
        clearError();
      } catch (error) {
        handleError(error);
      }
    });
  });
  container.querySelectorAll('button[data-action="edit"]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      const idx = Number(event.currentTarget.dataset.index);
      const line = state.plots.lines[idx];
      if (!line) return;
      openPlotLineDialog({ index: idx, defaults: { ...line } });
    });
  });
}

function openPlotLineDialog({ index = null, defaults = {} } = {}) {
  if (!state.metadata || !state.plots) return;
  const container = document.getElementById('plot-line-form');
  container.innerHTML = '';
  container.classList.remove('hidden');

  const totalPanels = Math.max(1, state.plots.rows * state.plots.cols);
  const maxPanelIndex = totalPanels - 1;
  const heading = document.createElement('h3');
  heading.textContent = index === null ? 'Add Plot Line' : 'Edit Plot Line';
  container.appendChild(heading);

  const form = document.createElement('form');
  form.className = 'grid-form';

  const panelLabel = document.createElement('label');
  panelLabel.textContent = `Panel index (0-${maxPanelIndex})`;
  const panelInput = document.createElement('input');
  panelInput.type = 'number';
  panelInput.min = '0';
  panelInput.max = String(maxPanelIndex);
  const defaultPanelRaw =
    typeof defaults.panel === 'number' ? defaults.panel : Number(defaults.panel ?? 0);
  const defaultPanel = Number.isFinite(defaultPanelRaw)
    ? Math.min(maxPanelIndex, Math.max(0, Math.floor(defaultPanelRaw)))
    : 0;
  panelInput.value = defaultPanel;
  panelLabel.appendChild(panelInput);
  form.appendChild(panelLabel);

  const tagLabel = document.createElement('label');
  tagLabel.textContent = 'Tag';
  const tagSelect = document.createElement('select');
  const tagOptions = Array.from(
    new Set([...(state.metadata.usable_tags || []), ...(state.metadata.setpoint_tags || [])])
  ).sort((a, b) => a.localeCompare(b));
  const hasTags = tagOptions.length > 0;
  if (hasTags) {
    tagOptions.forEach((tag) => {
      const option = document.createElement('option');
      option.value = tag;
      option.textContent = tag;
      tagSelect.appendChild(option);
    });
    const defaultTag = defaults.tag && tagOptions.includes(defaults.tag) ? defaults.tag : tagOptions[0];
    tagSelect.value = defaultTag;
  } else {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No tags available';
    tagSelect.appendChild(option);
    tagSelect.disabled = true;
  }
  tagLabel.appendChild(tagSelect);
  form.appendChild(tagLabel);

  const labelLabel = document.createElement('label');
  labelLabel.textContent = 'Label (optional)';
  const labelInput = document.createElement('input');
  labelInput.type = 'text';
  labelInput.value = defaults.label ?? '';
  labelLabel.appendChild(labelInput);
  form.appendChild(labelLabel);

  const colorLabel = document.createElement('label');
  colorLabel.textContent = 'Color (optional)';
  const colorInput = document.createElement('input');
  colorInput.type = 'text';
  colorInput.value = defaults.color ?? '';
  colorLabel.appendChild(colorInput);
  form.appendChild(colorLabel);

  const styleLabel = document.createElement('label');
  styleLabel.textContent = 'Line style (optional)';
  const styleInput = document.createElement('input');
  styleInput.type = 'text';
  styleInput.value = defaults.style ?? '';
  styleLabel.appendChild(styleInput);
  form.appendChild(styleLabel);

  const controls = document.createElement('div');
  controls.className = 'row';
  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.textContent = 'Save';
  if (!hasTags) {
    submit.disabled = true;
  }
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel';
  cancel.addEventListener('click', () => {
    container.classList.add('hidden');
    container.innerHTML = '';
  });
  controls.appendChild(submit);
  controls.appendChild(cancel);
  form.appendChild(controls);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!tagSelect.value) {
      showError('Select a tag for the plot line.');
      return;
    }
    const parsedPanel = Number(panelInput.value);
    const safePanel = Number.isFinite(parsedPanel) ? Math.floor(parsedPanel) : 0;
    const panelIndex = Math.min(maxPanelIndex, Math.max(0, safePanel));
    const line = {
      panel: panelIndex,
      tag: tagSelect.value,
    };
    const label = labelInput.value.trim();
    const color = colorInput.value.trim();
    const style = styleInput.value.trim();
    if (label) line.label = label;
    if (color) line.color = color;
    if (style) line.style = style;

    const updatedLines =
      index === null
        ? [...state.plots.lines, line]
        : state.plots.lines.map((existing, i) => (i === index ? line : existing));

    try {
      await updatePlotLayout({ ...state.plots, lines: updatedLines });
      container.classList.add('hidden');
      container.innerHTML = '';
      clearError();
    } catch (error) {
      handleError(error);
    }
  });

  if (!hasTags) {
    const helper = document.createElement('p');
    helper.textContent = 'No plot tags are available. Add sensors, calculations, or controllers first.';
    form.insertBefore(helper, controls);
  }

  container.appendChild(form);
}

function renderRunResult(result) {
  const status = document.getElementById('run-status');
  if (!result) {
    status.textContent = '';
    return;
  }
  status.textContent = `Simulation completed. Current time: ${result.time.toFixed(3)}.`;
  const plotOutput = document.getElementById('plot-output');
  if (result.figure) {
    const img = document.getElementById('plot-image');
    img.src = result.figure;
    plotOutput.classList.remove('hidden');
  } else {
    plotOutput.classList.add('hidden');
  }
}

function choiceOptions(typeLabel, fieldName) {
  if (!state.metadata) return [];
  if (fieldName === 'measurement_tag') {
    return state.metadata.measurables.map((m) => m.tag);
  }
  if (fieldName === 'mv_tag') {
    return state.metadata.control_elements || [];
  }
  if (fieldName === 'cv_tag') {
    const usable = new Set([...(state.metadata.usable_tags || [])]);
    if (usable.size === 0) {
      state.metadata.measurables.forEach((m) => usable.add(m.tag));
    }
    return Array.from(usable);
  }
  if (typeLabel === 'enum') {
    return ['AUTO', 'CASCADE', 'TRACKING'];
  }
  return null;
}

function buildFieldInput(field, defaults = {}) {
  const wrapper = document.createElement('label');
  wrapper.dataset.field = field.name;
  wrapper.dataset.type = field.type;
  wrapper.className = 'field-label';

  const header = document.createElement('div');
  header.className = 'field-label__header';
  const title = document.createElement('span');
  title.className = 'field-title';
  title.textContent = field.name;
  header.appendChild(title);
  if (field.description) {
    const info = document.createElement('span');
    info.className = 'info-icon';
    info.textContent = 'i';
    info.title = field.description;
    info.tabIndex = 0;
    header.appendChild(info);
  }
  wrapper.appendChild(header);

  const defaultValue = defaults[field.name] ?? field.default ?? '';
  const options = choiceOptions(field.type, field.name);

  const buildGroup = (labelText, input) => {
    const group = document.createElement('div');
    group.className = 'field-input-group';
    const caption = document.createElement('span');
    caption.textContent = labelText;
    group.appendChild(caption);
    group.appendChild(input);
    return group;
  };

  if (field.type === 'quantity') {
    const row = document.createElement('div');
    row.className = 'row';
    const valueInput = document.createElement('input');
    valueInput.type = 'number';
    valueInput.step = 'any';
    valueInput.value = defaultValue?.value ?? '';
    valueInput.dataset.role = 'value';
    const unitInput = document.createElement('input');
    unitInput.type = 'text';
    unitInput.value = defaultValue?.unit ?? '';
    unitInput.dataset.role = 'unit';
    row.appendChild(buildGroup('Value', valueInput));
    const unitGroup = buildGroup('Unit', unitInput);
    unitGroup.appendChild(createUnitHint());
    row.appendChild(unitGroup);
    wrapper.appendChild(row);
    return wrapper;
  }

  if (field.type === 'unit') {
    const input = document.createElement('input');
    input.type = 'text';
    input.dataset.role = 'unit';
    input.value = defaultValue ?? '';
    const group = buildGroup('Unit', input);
    group.appendChild(createUnitHint());
    wrapper.appendChild(group);
    return wrapper;
  }

  if (field.type === 'quantity_range') {
    const lower = Array.isArray(defaultValue)
      ? defaultValue[0] || {}
      : defaultValue?.lower || {};
    const upper = Array.isArray(defaultValue)
      ? defaultValue[1] || {}
      : defaultValue?.upper || {};
    const row = document.createElement('div');
    row.className = 'row';
    const lowerValueInput = document.createElement('input');
    lowerValueInput.type = 'number';
    lowerValueInput.step = 'any';
    lowerValueInput.value = lower.value ?? '';
    lowerValueInput.dataset.role = 'lower-value';
    const lowerUnitInput = document.createElement('input');
    lowerUnitInput.type = 'text';
    lowerUnitInput.dataset.role = 'lower-unit';
    lowerUnitInput.value = lower.unit ?? '';
    const upperValueInput = document.createElement('input');
    upperValueInput.type = 'number';
    upperValueInput.step = 'any';
    upperValueInput.value = upper.value ?? '';
    upperValueInput.dataset.role = 'upper-value';
    const upperUnitInput = document.createElement('input');
    upperUnitInput.type = 'text';
    upperUnitInput.dataset.role = 'upper-unit';
    upperUnitInput.value = upper.unit ?? '';
    row.appendChild(buildGroup('Lower value', lowerValueInput));
    const lowerUnitGroup = buildGroup('Lower unit', lowerUnitInput);
    lowerUnitGroup.appendChild(createUnitHint('lower-unit-hint'));
    row.appendChild(lowerUnitGroup);
    row.appendChild(buildGroup('Upper value', upperValueInput));
    const upperUnitGroup = buildGroup('Upper unit', upperUnitInput);
    upperUnitGroup.appendChild(createUnitHint('upper-unit-hint'));
    row.appendChild(upperUnitGroup);
    wrapper.appendChild(row);
    return wrapper;
  }

  if (field.type === 'tuple[number]') {
    const current = Array.isArray(defaultValue) ? defaultValue : [];
    const row = document.createElement('div');
    row.className = 'row';
    const minInput = document.createElement('input');
    minInput.type = 'number';
    minInput.step = 'any';
    minInput.value = current[0] ?? '';
    minInput.dataset.role = 'min';
    const maxInput = document.createElement('input');
    maxInput.type = 'number';
    maxInput.step = 'any';
    maxInput.value = current[1] ?? '';
    maxInput.dataset.role = 'max';
    row.appendChild(buildGroup('Min', minInput));
    row.appendChild(buildGroup('Max', maxInput));
    wrapper.appendChild(row);
    return wrapper;
  }

  if (field.type === 'boolean') {
    const container = document.createElement('div');
    container.className = 'field-input-group';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = Boolean(defaultValue);
    input.dataset.role = 'boolean';
    container.appendChild(input);
    const caption = document.createElement('span');
    caption.textContent = 'Enabled';
    container.appendChild(caption);
    wrapper.appendChild(container);
    return wrapper;
  }

  if (options && options.length > 0) {
    const select = document.createElement('select');
    select.innerHTML = options.map((value) => `<option value="${value}">${value}</option>`).join('');
    if (defaultValue) {
      select.value = defaultValue;
    }
    wrapper.appendChild(select);
    return wrapper;
  }

  const input = document.createElement('input');
  input.type = field.type === 'number' || field.type === 'integer' ? 'number' : 'text';
  if (field.type === 'integer') {
    input.step = '1';
  }
  input.value = defaultValue ?? '';
  wrapper.appendChild(input);
  return wrapper;
}

function extractFieldValue(label) {
  const name = label.dataset.field;
  const type = label.dataset.type;
  if (type === 'quantity') {
    const value = label.querySelector('input[data-role="value"]').value;
    const unit = label.querySelector('input[data-role="unit"]').value;
    if (value === '') return null;
    return { value: Number(value), unit };
  }
  if (type === 'quantity_range') {
    const lowerValue = label.querySelector('input[data-role="lower-value"]').value;
    const lowerUnitElement = label.querySelector('[data-role="lower-unit"]');
    const lowerUnit = lowerUnitElement ? lowerUnitElement.value : '';
    const upperValue = label.querySelector('input[data-role="upper-value"]').value;
    const upperUnitElement = label.querySelector('[data-role="upper-unit"]');
    const upperUnit = upperUnitElement ? upperUnitElement.value : '';
    if (lowerValue === '' || upperValue === '') return null;
    return {
      lower: { value: Number(lowerValue), unit: lowerUnit },
      upper: { value: Number(upperValue), unit: upperUnit },
    };
  }
  if (type === 'tuple[number]') {
    const min = label.querySelector('input[data-role="min"]').value;
    const max = label.querySelector('input[data-role="max"]').value;
    if (min === '' || max === '') return null;
    return [Number(min), Number(max)];
  }
  if (type === 'boolean') {
    return label.querySelector('input[type="checkbox"]').checked;
  }
  const select = label.querySelector('select');
  if (select) {
    return select.value;
  }
  const input = label.querySelector('input');
  if (!input) return null;
  if (input.value === '') return null;
  if (type === 'number' || type === 'integer') {
    return Number(input.value);
  }
  return input.value;
}

function buildDynamicForm(container, typeList, submitHandler, options = {}) {
  container.innerHTML = '';
  container.classList.remove('hidden');
  const form = document.createElement('form');
  form.className = 'dynamic-form';
  const header = document.createElement('div');
  header.className = 'row';
  const typeLabel = document.createElement('label');
  typeLabel.textContent = 'Type';
  const select = document.createElement('select');
  select.innerHTML = typeList.map((item) => `<option value="${item.name}">${item.name}</option>`).join('');
  typeLabel.appendChild(select);
  header.appendChild(typeLabel);
  form.appendChild(header);

  const fieldsContainer = document.createElement('div');
  fieldsContainer.className = 'row';
  form.appendChild(fieldsContainer);

  let currentType = typeList[0];
  if (options.initialTypeName) {
    const selected = typeList.find((item) => item.name === options.initialTypeName);
    if (selected) {
      currentType = selected;
    }
  }
  select.value = currentType.name;
  if (options.lockType) {
    select.disabled = true;
  }

  function renderFields(defaults = {}) {
    fieldsContainer.innerHTML = '';
    currentType.fields.forEach((field) => {
      const skip = options.exclude?.includes(field.name);
      if (skip) return;
      const label = buildFieldInput(field, defaults);
      fieldsContainer.appendChild(label);
    });
    if (typeof options.onFieldsRendered === 'function') {
      options.onFieldsRendered({ container: fieldsContainer, type: currentType, defaults });
    }
  }

  renderFields(options.defaults || {});

  select.addEventListener('change', () => {
    currentType = typeList.find((item) => item.name === select.value) || currentType;
    renderFields(options.defaults || {});
    if (options.onTypeChange) {
      options.onTypeChange(currentType);
    }
  });

  const footer = document.createElement('div');
  footer.className = 'row';
  const submit = document.createElement('button');
  submit.type = 'submit';
  submit.textContent = 'Save';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel';
  cancel.addEventListener('click', () => {
    container.classList.add('hidden');
    container.innerHTML = '';
  });
  footer.appendChild(submit);
  footer.appendChild(cancel);
  form.appendChild(footer);

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const values = {};
    fieldsContainer.querySelectorAll('label[data-field]').forEach((label) => {
      const value = extractFieldValue(label);
      if (value !== null && value !== '') {
        values[label.dataset.field] = value;
      }
    });
    try {
      await submitHandler({ type: currentType.name, values });
      clearError();
    } catch (error) {
      handleError(error);
    }
  });

  container.appendChild(form);
  return { form, select };
}

function openSensorForm() {
  if (!state.metadata) return;
  const container = document.getElementById('sensor-form');
  const types = state.metadata.sensor_types || [];
  buildDynamicForm(
    container,
    types,
    async ({ type, values }) => {
      await fetchJSON('/api/sensors', {
        method: 'POST',
        body: JSON.stringify({ type, params: values }),
      });
      container.classList.add('hidden');
      container.innerHTML = '';
      await refreshSensors();
      await refreshMetadata();
      clearError();
    },
    {
      onFieldsRendered: ({ container: fieldsContainer, defaults }) => {
        setupSensorUnitOptions(fieldsContainer, defaults || {});
      },
    }
  );
}

function buildTrajectoryBuilder(root, defaults = {}) {
  const localState = {
    segments: Array.isArray(defaults.segments)
      ? defaults.segments.map((segment) => ({ ...segment }))
      : [],
  };
  const wrapper = document.createElement('div');
  wrapper.className = 'dialog';
  const y0Input = document.createElement('input');
  y0Input.type = 'number';
  y0Input.step = 'any';
  y0Input.value = defaults.y0 ?? 0;
  const unitInput = document.createElement('input');
  unitInput.type = 'text';
  unitInput.dataset.role = 'unit';
  unitInput.value = defaults.unit ?? '';
  wrapper.innerHTML = '<h3>Setpoint Trajectory</h3>';
  const row = document.createElement('div');
  row.className = 'row';
  const yLabel = document.createElement('label');
  yLabel.textContent = 'Initial value';
  yLabel.appendChild(y0Input);
  const uLabel = document.createElement('label');
  uLabel.textContent = 'Unit';
  uLabel.appendChild(unitInput);
  const unitHint = createUnitHint();
  uLabel.appendChild(unitHint);
  row.appendChild(yLabel);
  row.appendChild(uLabel);
  wrapper.appendChild(row);

  const segmentsContainer = document.createElement('div');
  segmentsContainer.className = 'item-list';
  wrapper.appendChild(segmentsContainer);

  const timeUnit = state.metadata?.time_unit || '';
  const durationLabel = `Duration${timeUnit ? ` (${timeUnit})` : ''}`;
  const dtLabel = `dt${timeUnit ? ` (${timeUnit})` : ''}`;

  function refreshSegments() {
    segmentsContainer.innerHTML = '';
    if (localState.segments.length === 0) {
      const empty = document.createElement('p');
      empty.textContent = 'No segments defined. The setpoint will remain constant.';
      segmentsContainer.appendChild(empty);
      return;
    }
    localState.segments.forEach((segment, index) => {
      const card = document.createElement('div');
      card.className = 'item-card';
      card.innerHTML = `
        <header><h4>${segment.type}</h4></header>
        <pre>${JSON.stringify(segment, null, 2)}</pre>
        <div class="actions"><button type="button" data-index="${index}">Remove</button></div>
      `;
      segmentsContainer.appendChild(card);
    });
    segmentsContainer.querySelectorAll('button[data-index]').forEach((btn) => {
      btn.addEventListener('click', (event) => {
        const idx = Number(event.currentTarget.dataset.index);
        localState.segments.splice(idx, 1);
        refreshSegments();
      });
    });
  }

  refreshSegments();

  const addButton = document.createElement('button');
  addButton.type = 'button';
  addButton.textContent = 'Add Segment';
  addButton.addEventListener('click', () => {
    const dialog = document.createElement('div');
    dialog.className = 'dialog';
    const form = document.createElement('form');
    form.className = 'grid-form';
    const typeLabel = document.createElement('label');
    typeLabel.textContent = 'Type';
    const select = document.createElement('select');
    select.innerHTML = `
      <option value="hold">Hold</option>
      <option value="step">Step</option>
      <option value="ramp">Ramp</option>
      <option value="random_walk">Random Walk</option>
    `;
    typeLabel.appendChild(select);
    form.appendChild(typeLabel);

    const valueInputs = document.createElement('div');
    valueInputs.className = 'row';
    form.appendChild(valueInputs);

    function updateInputs() {
      valueInputs.innerHTML = '';
      const type = select.value;
      if (type === 'hold') {
        valueInputs.innerHTML = `
          <label>${durationLabel}<input name="duration" type="number" step="any" value="0" /></label>
          <label>Value<input name="value" type="number" step="any" /></label>
        `;
      } else if (type === 'step') {
        valueInputs.innerHTML = '<label>Magnitude<input name="magnitude" type="number" step="any" value="0" /></label>';
      } else if (type === 'ramp') {
        valueInputs.innerHTML = `
          <label>Magnitude<input name="magnitude" type="number" step="any" value="0" /></label>
          <label>${durationLabel}<input name="duration" type="number" step="any" value="0" /></label>
        `;
      } else {
        valueInputs.innerHTML = `
          <label>Std<input name="std" type="number" step="any" value="0.1" /></label>
          <label>${durationLabel}<input name="duration" type="number" step="any" value="1" /></label>
          <label>${dtLabel}<input name="dt" type="number" step="any" value="1" /></label>
        `;
      }
    }

    updateInputs();
    select.addEventListener('change', updateInputs);

    const controls = document.createElement('div');
    controls.className = 'row';
    const submit = document.createElement('button');
    submit.type = 'submit';
    submit.textContent = 'Add';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = 'Cancel';
    cancel.addEventListener('click', () => {
      dialog.remove();
    });
    controls.appendChild(submit);
    controls.appendChild(cancel);
    form.appendChild(controls);

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const segment = { type: select.value };
      data.forEach((value, key) => {
        if (value !== '') {
          segment[key] = Number(value);
        }
      });
      localState.segments.push(segment);
      dialog.remove();
      refreshSegments();
    });

    dialog.appendChild(form);
    wrapper.appendChild(dialog);
  });
  wrapper.appendChild(addButton);

  root.appendChild(wrapper);
  applyUnitMetadataToInput(
    unitInput,
    unitHint,
    normalizeUnitMetadata({ unit: defaults.unit ?? '', unit_aliases: [] }),
    defaults.unit ?? '',
    false
  );

  return {
    getTrajectory() {
      return {
        y0: Number(y0Input.value || 0),
        unit: unitInput.value || '',
        segments: localState.segments.map((segment) => ({ ...segment })),
      };
    },
    setUnitReference(metadata, preferredUnit, forceBase = false) {
      const normalized = normalizeUnitMetadata(metadata);
      const fallback =
        typeof preferredUnit === 'string' ? preferredUnit : defaults.unit ?? normalized.unit;
      applyUnitMetadataToInput(unitInput, unitHint, normalized, fallback, forceBase);
    },
  };
}

function openControllerForm({ defaults = {}, parentId = null } = {}) {
  if (!state.metadata) return;
  const container = document.getElementById('controller-form');
  const types = state.metadata.controller_types || [];
  if (types.length === 0) {
    container.innerHTML = '<p>No controller types available.</p>';
    container.classList.remove('hidden');
    return;
  }
  const defaultsWithUnits = { ...defaults };
  if (defaultsWithUnits.cv_tag && !defaultsWithUnits.sp_unit) {
    const info = getUsableTagInfo(defaultsWithUnits.cv_tag);
    if (info && typeof info.unit === 'string') {
      defaultsWithUnits.sp_unit = info.unit;
    }
  }
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'dialog';
  const inner = document.createElement('div');
  wrapper.appendChild(inner);
  container.appendChild(wrapper);

  let trajectoryBuilder = null;
  let pendingTrajectoryUnits = null;

  const { form, select } = buildDynamicForm(inner, types, async ({ type, values }) => {
    const trajectory = trajectoryBuilder ? trajectoryBuilder.getTrajectory() : { y0: 0, unit: '', segments: [] };
    await fetchJSON('/api/controllers', {
      method: 'POST',
      body: JSON.stringify({ type, params: values, trajectory, parent_id: parentId }),
    });
    container.classList.add('hidden');
    container.innerHTML = '';
    await refreshControllers();
    await refreshMetadata();
    clearError();
  }, {
    defaults: defaultsWithUnits,
    exclude: ['sp_trajectory', 'cascade_controller'],
    onTypeChange: () => {
      // no-op; trajectory unaffected
    },
    onFieldsRendered: ({ container: fieldsContainer }) => {
      setupMvRangeUnits(fieldsContainer, defaultsWithUnits);
      const cvTagLabel = fieldsContainer.querySelector('label[data-field="cv_tag"]');
      const cvSelect = cvTagLabel ? cvTagLabel.querySelector('select') : null;
      const cvInput = cvTagLabel ? cvTagLabel.querySelector('input') : null;

      const updateTrajectoryUnits = (preferredUnit) => {
        let tag = '';
        if (cvSelect) {
          tag = cvSelect.value;
        } else if (cvInput) {
          tag = cvInput.value;
        } else if (typeof defaultsWithUnits.cv_tag === 'string') {
          tag = defaultsWithUnits.cv_tag;
        }
        const info = getUsableTagInfo(tag);
        const metadata = normalizeUnitMetadata(info);
        const desired = typeof preferredUnit === 'string' ? preferredUnit : '';
        const forceBase = preferredUnit === null;
        if (trajectoryBuilder) {
          trajectoryBuilder.setUnitReference(metadata, desired, forceBase);
        } else {
          pendingTrajectoryUnits = { metadata, preferred: desired, forceBase };
        }
      };

      updateTrajectoryUnits(defaultsWithUnits.sp_unit || '');

      if (cvSelect) {
        cvSelect.addEventListener('change', () => updateTrajectoryUnits(null));
      } else if (cvInput) {
        cvInput.addEventListener('blur', () => updateTrajectoryUnits(null));
      }
    },
  });

  trajectoryBuilder = buildTrajectoryBuilder(wrapper, {
    y0: defaultsWithUnits.sp_y0 ?? 0,
    unit: defaultsWithUnits.sp_unit ?? '',
    segments: [],
  });

  if (pendingTrajectoryUnits) {
    trajectoryBuilder.setUnitReference(
      pendingTrajectoryUnits.metadata,
      pendingTrajectoryUnits.preferred,
      pendingTrajectoryUnits.forceBase
    );
    pendingTrajectoryUnits = null;
  }

  container.classList.remove('hidden');
}

function openControllerEditor(controller) {
  if (!state.metadata) return;
  const container = document.getElementById('controller-form');
  const types = state.metadata.controller_types || [];
  if (types.length === 0) {
    container.innerHTML = '<div class="dialog"><p>No controller types available.</p></div>';
    container.classList.remove('hidden');
    return;
  }
  const availableType = types.find((item) => item.name === controller.type);
  if (!availableType) {
    container.innerHTML = `<div class="dialog"><p>The controller type <strong>${controller.type}</strong> is no longer available.</p></div>`;
    container.classList.remove('hidden');
    return;
  }

  const defaults = { ...(controller.params || {}) };
  const { form } = buildDynamicForm(
    container,
    types,
    async ({ values }) => {
      await fetchJSON(`/api/controllers/${controller.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ params: values }),
      });
      container.classList.add('hidden');
      container.innerHTML = '';
      await refreshControllers();
      await refreshMetadata();
      clearError();
    },
    {
      defaults,
      exclude: ['sp_trajectory', 'cascade_controller'],
      initialTypeName: controller.type,
      lockType: true,
      onFieldsRendered: ({ container: fieldsContainer }) => {
        setupMvRangeUnits(fieldsContainer, defaults);
      },
    }
  );

  const heading = document.createElement('h3');
  heading.textContent = `Edit ${controller.type} Settings`;
  form.parentElement?.insertBefore(heading, form);
}

function openTrajectoryEditor(controller) {
  const container = document.getElementById('controller-form');
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'dialog';
  const heading = document.createElement('h3');
  heading.textContent = `Edit ${controller.type} Setpoint`;
  wrapper.appendChild(heading);

  const trajectoryDefaults = controller.trajectory || { y0: 0, unit: '', segments: [] };
  const builder = buildTrajectoryBuilder(wrapper, {
    y0: trajectoryDefaults.y0 ?? 0,
    unit: trajectoryDefaults.unit ?? '',
    segments: trajectoryDefaults.segments || [],
  });

  const cvTag = controller.params?.cv_tag;
  const cvInfo = getUsableTagInfo(typeof cvTag === 'string' ? cvTag : '');
  const metadata = normalizeUnitMetadata(cvInfo);
  const preferredUnit =
    typeof trajectoryDefaults.unit === 'string' && trajectoryDefaults.unit.trim() !== ''
      ? trajectoryDefaults.unit
      : metadata.unit;
  builder.setUnitReference(metadata, preferredUnit, false);

  const footer = document.createElement('div');
  footer.className = 'row';
  const save = document.createElement('button');
  save.type = 'button';
  save.textContent = 'Save';
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = 'Cancel';

  cancel.addEventListener('click', () => {
    container.classList.add('hidden');
    container.innerHTML = '';
  });

  save.addEventListener('click', async () => {
    try {
      const trajectory = builder.getTrajectory();
      await fetchJSON(`/api/controllers/${controller.id}/trajectory`, {
        method: 'PATCH',
        body: JSON.stringify({ trajectory }),
      });
      container.classList.add('hidden');
      container.innerHTML = '';
      await refreshControllers();
      await refreshMetadata();
      clearError();
    } catch (error) {
      handleError(error);
    }
  });

  footer.appendChild(save);
  footer.appendChild(cancel);
  wrapper.appendChild(footer);

  container.appendChild(wrapper);
  container.classList.remove('hidden');
}

function openCalculationForm() {
  if (!state.metadata) return;
  const container = document.getElementById('calculation-form');
  const types = state.metadata.calculation_types || [];
  if (types.length === 0) {
    container.innerHTML = '<p>No calculation classes available. Upload a module first.</p>';
    container.classList.remove('hidden');
    return;
  }
  buildDynamicForm(container, types, async ({ type, values }) => {
    await fetchJSON('/api/calculations', {
      method: 'POST',
      body: JSON.stringify({ type, params: values }),
    });
    container.classList.add('hidden');
    container.innerHTML = '';
    await refreshCalculations();
    await refreshMetadata();
    clearError();
  });
}

async function refreshMetadata() {
  const data = await fetchJSON('/api/metadata');
  state.metadata = data;
  renderMeasurables();
  renderMessages(data.messages || []);
}

async function refreshSensors() {
  state.sensors = await fetchJSON('/api/sensors');
  renderSensorList();
}

async function refreshControllers() {
  state.controllers = await fetchJSON('/api/controllers');
  renderControllerList();
}

async function refreshCalculations() {
  state.calculations = await fetchJSON('/api/calculations');
  renderCalculationList();
}

async function refreshPlots() {
  state.plots = await fetchJSON('/api/plots');
  renderPlotLayout();
}

async function updatePlotLayout(layout) {
  state.plots = await fetchJSON('/api/plots', {
    method: 'POST',
    body: JSON.stringify(layout),
  });
  renderPlotLayout();
}

function initEventHandlers() {
  document.getElementById('add-sensor').addEventListener('click', () => openSensorForm());
  document.getElementById('add-controller').addEventListener('click', () => openControllerForm());
  document.getElementById('add-calculation').addEventListener('click', () => openCalculationForm());

  document.getElementById('plot-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const rows = Number(event.target.rows.value || 1);
    const cols = Number(event.target.cols.value || 1);
    try {
      await updatePlotLayout({ rows, cols, lines: state.plots ? state.plots.lines : [] });
      clearError();
    } catch (error) {
      handleError(error);
    }
  });

  document.getElementById('add-plot-line').addEventListener('click', () => {
    openPlotLineDialog();
  });

  document.getElementById('run-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const value = event.target.duration_value.value;
    const unit = event.target.duration_unit.value;
    let payload = {};
    if (value) {
      payload.duration = { value: Number(value), unit: unit || 's' };
    }
    try {
      const result = await fetchJSON('/api/run', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      renderRunResult(result);
      renderMessages(result.messages || []);
      await refreshMetadata();
      clearError();
    } catch (error) {
      handleError(error);
    }
  });

  document.getElementById('calculation-upload').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    try {
      const response = await fetch('/api/calculations/upload', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || 'Upload failed');
      }
      await refreshMetadata();
      clearError();
    } catch (error) {
      handleError(error);
    }
  });
}

async function initialize() {
  await refreshMetadata();
  await Promise.all([refreshSensors(), refreshControllers(), refreshCalculations(), refreshPlots()]);
}

initEventHandlers();
initialize().catch((error) => {
  handleError(error);
});

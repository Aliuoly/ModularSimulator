const numberFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 });
const timeFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

const charts = {
  b: null,
  flow: null,
  volume: null,
};

const DEFAULT_TIME_WINDOW_MINUTES = 30;
const MAX_POINTS_PER_SERIES = 400;
const SERVER_POINTS_PER_MINUTE = 12;
const MIN_SERVER_POINTS = 150;
const MAX_SERVER_POINTS = 1200;

let timeWindowMinutes = DEFAULT_TIME_WINDOW_MINUTES;
let latestTrendPayload = null;

function computeRange(datasets) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  datasets.forEach((dataset) => {
    (dataset.data || []).forEach((point) => {
      const value = Number(point.y);
      if (!Number.isFinite(value)) {
        return;
      }
      if (value < min) min = value;
      if (value > max) max = value;
    });
  });

  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return null;
  }

  if (min === max) {
    const offset = Math.max(Math.abs(min) * 0.1, 0.1);
    return { min: min - offset, max: max + offset };
  }

  const padding = Math.max((max - min) * 0.1, 1e-6);
  return { min: min - padding, max: max + padding };
}

function applyRange(chart, datasets) {
  const range = computeRange(datasets);
  if (range) {
    chart.options.scales.y.min = range.min;
    chart.options.scales.y.max = range.max;
  } else {
    chart.options.scales.y.min = undefined;
    chart.options.scales.y.max = undefined;
  }
}

function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return numberFormatter.format(value);
}

function toScalar(value) {
  if (Array.isArray(value)) {
    if (!value.length) {
      return null;
    }
    return toScalar(value[value.length - 1]);
  }
  if (value && typeof value === "object") {
    if ("value" in value) {
      return toScalar(value.value);
    }
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function decimate(points, maxPoints) {
  if (!Array.isArray(points) || points.length <= maxPoints) {
    return points || [];
  }

  const decimated = [];
  const step = (points.length - 1) / (maxPoints - 1);

  for (let i = 0; i < maxPoints; i += 1) {
    const index = Math.floor(i * step);
    const chosen = points[index];
    if (!chosen) {
      continue;
    }
    if (!decimated.length) {
      decimated.push(chosen);
      continue;
    }
    const last = decimated[decimated.length - 1];
    if (last.x !== chosen.x || last.y !== chosen.y) {
      decimated.push(chosen);
    }
  }

  const lastPoint = points[points.length - 1];
  const lastDecimated = decimated[decimated.length - 1];
  if (lastPoint && lastDecimated && (lastPoint.x !== lastDecimated.x || lastPoint.y !== lastDecimated.y)) {
    decimated.push(lastPoint);
  }

  return decimated;
}

function buildPoints(history, windowMinutes) {
  if (!Array.isArray(history)) {
    return [];
  }

  const points = [];
  let lastX = null;
  let lastY = null;
  let maxX = Number.NEGATIVE_INFINITY;

  history.forEach((point) => {
    if (point && point.ok === false) {
      return;
    }

    const rawTime = Number(point?.time);
    const rawValue = toScalar(point?.value);
    if (!Number.isFinite(rawTime) || rawValue === null) {
      return;
    }

    const x = rawTime / 60.0;
    const y = rawValue;

    if (
      lastX !== null &&
      lastY !== null &&
      Math.abs(x - lastX) < 1e-9 &&
      Math.abs(y - lastY) < 1e-9
    ) {
      return;
    }

    points.push({ x, y });
    lastX = x;
    lastY = y;
    if (x > maxX) {
      maxX = x;
    }
  });

  if (!points.length) {
    return points;
  }

  let filtered = points;
  if (Number.isFinite(windowMinutes) && windowMinutes > 0 && Number.isFinite(maxX)) {
    const minX = maxX - windowMinutes;
    filtered = points.filter((point) => point.x >= minX);
  }

  return decimate(filtered, MAX_POINTS_PER_SERIES);
}

function applyTimeRange(chart, currentTimeMinutes) {
  if (!chart) {
    return;
  }
  const xScale = chart.options.scales.x;
  if (!xScale) {
    return;
  }

  if (Number.isFinite(currentTimeMinutes)) {
    const window = Number.isFinite(timeWindowMinutes) && timeWindowMinutes > 0 ? timeWindowMinutes : null;
    const min = window ? Math.max(currentTimeMinutes - window, 0) : undefined;
    if (min !== undefined) {
      xScale.min = min;
    } else {
      delete xScale.min;
    }
    xScale.max = currentTimeMinutes;
  } else {
    delete xScale.min;
    delete xScale.max;
  }
}

function initCharts() {
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "nearest", intersect: false },
    parsing: false,
    animation: false,
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Time (minutes)" },
      },
    },
    plugins: {
      legend: { position: "top" },
      decimation: {
        enabled: true,
        algorithm: "lttb",
        samples: MAX_POINTS_PER_SERIES,
      },
    },
  };

  charts.b = new Chart(document.getElementById("chart-b"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Concentration" },
        },
      },
    },
  });

  charts.flow = new Chart(document.getElementById("chart-flow"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Flow Rate" },
        },
      },
    },
  });

  charts.volume = new Chart(document.getElementById("chart-volume"), {
    type: "line",
    data: { datasets: [] },
    options: {
      ...baseOptions,
      scales: {
        ...baseOptions.scales,
        y: {
          title: { display: true, text: "Volume" },
        },
      },
    },
  });
}

function updateCharts(data) {
  if (!charts.b || !charts.flow || !charts.volume) {
    return;
  }

  latestTrendPayload = data;

  const sensorData = data.sensors || {};
  const setpoints = data.setpoints || {};
  const manipulated = data.manipulated || {};

  const rawTime = Number(data?.time);
  const currentTimeMinutes = Number.isFinite(rawTime) ? rawTime / 60.0 : undefined;

  const bSensor = sensorData.B ? buildPoints(sensorData.B.data, timeWindowMinutes) : [];
  const bSetpoint = setpoints.B ? buildPoints(setpoints.B.data, timeWindowMinutes) : [];

  const bDatasets = [
    {
      label: "B (sensor)",
      data: bSensor,
      borderColor: "#2563eb",
      backgroundColor: "rgba(37, 99, 235, 0.2)",
      tension: 0.2,
      spanGaps: true,
      pointRadius: 2,
      pointHoverRadius: 4,
    },
    {
      label: "B Setpoint",
      data: bSetpoint,
      borderColor: "#f97316",
      borderDash: [6, 4],
      backgroundColor: "rgba(249, 115, 22, 0.2)",
      tension: 0.2,
    },
  ];
  charts.b.data.datasets = bDatasets;
  applyRange(charts.b, bDatasets);
  applyTimeRange(charts.b, currentTimeMinutes);
  charts.b.update("none");

  const fInSensor = sensorData.F_in ? buildPoints(sensorData.F_in.data, timeWindowMinutes) : [];
  const fOutSensor = sensorData.F_out ? buildPoints(sensorData.F_out.data, timeWindowMinutes) : [];
  const fInCommand = manipulated.F_in ? buildPoints(manipulated.F_in.data, timeWindowMinutes) : [];

  const flowDatasets = [
    {
      label: "F_in (sensor)",
      data: fInSensor,
      borderColor: "#16a34a",
      backgroundColor: "rgba(22, 163, 74, 0.2)",
      tension: 0.2,
    },
    {
      label: "F_out",
      data: fOutSensor,
      borderColor: "#dc2626",
      backgroundColor: "rgba(220, 38, 38, 0.2)",
      tension: 0.2,
    },
    {
      label: "F_in (command)",
      data: fInCommand,
      borderColor: "#0ea5e9",
      borderDash: [4, 4],
      backgroundColor: "rgba(14, 165, 233, 0.2)",
      tension: 0.2,
    },
  ];
  charts.flow.data.datasets = flowDatasets;
  applyRange(charts.flow, flowDatasets);
  applyTimeRange(charts.flow, currentTimeMinutes);
  charts.flow.update("none");

  const volumeSensor = sensorData.V ? buildPoints(sensorData.V.data, timeWindowMinutes) : [];
  const volumeDatasets = [
    {
      label: "V (sensor)",
      data: volumeSensor,
      borderColor: "#9333ea",
      backgroundColor: "rgba(147, 51, 234, 0.2)",
      tension: 0.2,
    },
  ];
  charts.volume.data.datasets = volumeDatasets;
  applyRange(charts.volume, volumeDatasets);
  applyTimeRange(charts.volume, currentTimeMinutes);
  charts.volume.update("none");
}

function updateTimeAndSpeed(timeSeconds, speed) {
  const timeElement = document.getElementById("sim-time");
  const coercedTime = Number(timeSeconds);
  if (timeElement && Number.isFinite(coercedTime)) {
    timeElement.textContent = timeFormatter.format(coercedTime / 60.0);
  }
  const speedElement = document.getElementById("speed-factor");
  const coercedSpeed = Number(speed);
  if (speedElement && Number.isFinite(coercedSpeed)) {
    speedElement.textContent = numberFormatter.format(coercedSpeed);
  }
}

function showMessage(text, type = "info") {
  const container = document.getElementById("message");
  if (!container) return;
  container.textContent = text;
  container.className = "message " + type;
  container.classList.add("visible");
  setTimeout(() => {
    container.classList.remove("visible");
  }, 4000);
}

function buildControllerCard(controller) {
  const card = document.createElement("div");
  card.className = "controller-card";
  card.dataset.cvTag = controller.cv_tag;

  const header = document.createElement("div");
  header.className = "controller-card__header";
  header.innerHTML = `<h3>${controller.cv_tag}</h3>`;
  card.appendChild(header);

  const modeWrapper = document.createElement("div");
  modeWrapper.className = "controller-card__row";
  const modeLabel = document.createElement("label");
  modeLabel.textContent = "Mode";
  const modeSelect = document.createElement("select");
  controller.available_modes.forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode;
    if (mode === controller.mode) {
      option.selected = true;
    }
    modeSelect.appendChild(option);
  });
  modeSelect.addEventListener("change", (event) => {
    const newMode = event.target.value;
    updateControllerMode(controller.cv_tag, newMode);
  });
  modeWrapper.appendChild(modeLabel);
  modeWrapper.appendChild(modeSelect);
  card.appendChild(modeWrapper);

  const pvRow = document.createElement("div");
  pvRow.className = "controller-card__row controller-card__row--status";
  const pvStatus = controller.pv.ok ? "OK" : "Fault";
  pvRow.innerHTML = `PV: <strong>${formatValue(controller.pv.value)}</strong> ${controller.pv.unit} <span class="status ${controller.pv.ok ? "status--ok" : "status--bad"}">${pvStatus}</span>`;
  card.appendChild(pvRow);

  const mvRow = document.createElement("div");
  mvRow.className = "controller-card__row controller-card__row--status";
  mvRow.innerHTML = `MV (${controller.manipulated.tag}): <strong>${formatValue(controller.manipulated.value)}</strong> ${controller.manipulated.unit}`;
  card.appendChild(mvRow);

  const form = document.createElement("form");
  form.className = "controller-card__setpoint";
  const setpointLabel = document.createElement("label");
  setpointLabel.textContent = `Setpoint (${controller.setpoint.unit})`;
  const setpointInput = document.createElement("input");
  setpointInput.type = "number";
  setpointInput.step = "0.01";
  setpointInput.value = controller.setpoint.value ?? "";
  const applyButton = document.createElement("button");
  applyButton.type = "submit";
  applyButton.textContent = "Apply";

  const inAuto = controller.mode === "AUTO";
  setpointInput.disabled = !inAuto;
  applyButton.disabled = !inAuto;
  if (!inAuto) {
    applyButton.title = "Setpoint changes available only in AUTO mode.";
  }

  form.appendChild(setpointLabel);
  form.appendChild(setpointInput);
  form.appendChild(applyButton);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = parseFloat(setpointInput.value);
    if (Number.isNaN(value)) {
      showMessage("Please enter a numeric setpoint value.", "error");
      return;
    }
    updateControllerSetpoint(controller.cv_tag, value);
  });
  card.appendChild(form);

  if (controller.children && controller.children.length) {
    const childContainer = document.createElement("div");
    childContainer.className = "controller-card__children";
    controller.children.forEach((child) => {
      childContainer.appendChild(buildControllerCard(child));
    });
    card.appendChild(childContainer);
  }

  return card;
}

function renderControllers(payload) {
  const container = document.getElementById("controllers");
  if (!container) return;
  container.innerHTML = "";
  const controllers = payload.controllers || [];
  if (!controllers.length) {
    container.innerHTML = "<p>No controllers configured.</p>";
    return;
  }
  controllers.forEach((controller) => {
    container.appendChild(buildControllerCard(controller));
  });
}

async function fetchControllers() {
  try {
    const response = await fetch("/api/controllers");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
  } catch (error) {
    console.error("Failed to load controller metadata", error);
    showMessage(`Unable to load controllers: ${error}`, "error");
  }
}

function trendRequestPoints() {
  if (Number.isFinite(timeWindowMinutes) && timeWindowMinutes > 0) {
    const desired = Math.round(timeWindowMinutes * SERVER_POINTS_PER_MINUTE);
    return Math.max(MIN_SERVER_POINTS, Math.min(MAX_SERVER_POINTS, desired));
  }
  return MIN_SERVER_POINTS;
}

async function fetchTrends() {
  try {
    const points = trendRequestPoints();
    const response = await fetch(`/api/trends?points=${points}`);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    updateCharts(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
  } catch (error) {
    console.error("Failed to load trend data", error);
  }
}

async function updateControllerMode(cvTag, mode) {
  try {
    const response = await fetch(`/api/controllers/${encodeURIComponent(cvTag)}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
    showMessage(`Updated ${cvTag} mode to ${mode}.`, "success");
  } catch (error) {
    console.error("Failed to update controller mode", error);
    showMessage(`Unable to change mode: ${error}`, "error");
    fetchControllers();
  }
}

async function updateControllerSetpoint(cvTag, value) {
  try {
    const response = await fetch(`/api/controllers/${encodeURIComponent(cvTag)}/setpoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ setpoint: value }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    renderControllers(payload);
    updateTimeAndSpeed(payload.time, payload.speed);
    showMessage(`Setpoint for ${cvTag} updated.`, "success");
  } catch (error) {
    console.error("Failed to update setpoint", error);
    showMessage(`Unable to change setpoint: ${error}`, "error");
    fetchControllers();
  }
}

async function onSpeedSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("speed-input");
  if (!input) return;
  const value = parseFloat(input.value);
  if (Number.isNaN(value) || value <= 0) {
    showMessage("Speed factor must be a positive number.", "error");
    return;
  }
  try {
    const response = await fetch("/api/speed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ factor: value }),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const payload = await response.json();
    document.getElementById("speed-factor").textContent = numberFormatter.format(payload.speed);
    showMessage(`Simulation speed set to ${numberFormatter.format(payload.speed)}×`, "success");
  } catch (error) {
    console.error("Failed to update speed", error);
    showMessage(`Unable to change speed: ${error}`, "error");
  }
}

function setupEventHandlers() {
  const speedForm = document.getElementById("speed-form");
  if (speedForm) {
    speedForm.addEventListener("submit", onSpeedSubmit);
  }

  const timeWindowForm = document.getElementById("time-window-form");
  if (timeWindowForm) {
    timeWindowForm.addEventListener("submit", onTimeWindowSubmit);
  }
}

async function bootstrap() {
  initCharts();
  setupEventHandlers();
  const timeWindowInput = document.getElementById("time-window-input");
  if (timeWindowInput) {
    timeWindowInput.value = String(DEFAULT_TIME_WINDOW_MINUTES);
  }
  await fetchControllers();
  await fetchTrends();
  setInterval(fetchTrends, 1500);
}

window.addEventListener("DOMContentLoaded", bootstrap);

function onTimeWindowSubmit(event) {
  event.preventDefault();
  const input = document.getElementById("time-window-input");
  if (!input) {
    return;
  }

  const value = parseFloat(input.value);
  if (Number.isNaN(value) || value <= 0) {
    showMessage("Trend window must be a positive number of minutes.", "error");
    return;
  }

  timeWindowMinutes = value;
  showMessage(`Trend window set to ${numberFormatter.format(timeWindowMinutes)} minutes.`, "success");
  if (latestTrendPayload) {
    updateCharts(latestTrendPayload);
  }
  fetchTrends();
}

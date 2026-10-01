/**
 * frontend/app.js
 * Vanilla JS controller for Crisis Command: Leaflet map rendering,
 * real-time SSE streaming, interactive unit animation, approval modal orchestration,
 * coverage grid visualization, and multimodal incident ingestion.
 */

// ---------------------------------------------------------------------------
// Global Application State & Configuration
// ---------------------------------------------------------------------------
const APP_STATE = {
  map: null,
  center: [10.8050, 78.6856],
  zoom: 13,
  simSpeed: 30, // 1 real second = 30 simulated seconds (default match)

  // Map Layer Groups
  layers: {
    units: null,
    incidents: null,
    hazards: null,
    facilities: null,
    routes: null,
    ghostRoutes: null,
    coverage: null,
  },

  // Toggle Visibility Flags
  toggles: {
    coverage: true,
    ghost: true,
  },

  // Interactive Click Modes: null | 'add_incident' | 'draw_hazard'
  activeMode: null,

  // Stored Backend Snapshot
  store: {
    incidents: [],
    units: [],
    facilities: [],
    hazards: [],
    current_plan: null,
    previous_plan: null,
    pending_plan: null,
    events: [],
    coverage: null,
    staging_suggestions: []
  },

  // Active requestAnimationFrame unit animations { [unitId]: animState }
  animations: {},
  pendingThreadId: null
};

// Emoji mappings by resource type
const UNIT_ICONS = {
  ambulance: '🚑',
  rescue_team: '🚒',
  medical_unit: '🩺'
};

// ---------------------------------------------------------------------------
// Initialization & Map Setup
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  bindUIControls();
  initEventSource();
  fetchInitialState();
});

function initMap() {
  APP_STATE.map = L.map('map', {
    zoomControl: true,
    attributionControl: true
  }).setView(APP_STATE.center, APP_STATE.zoom);

  // CartoDB Dark Matter / OSM Tile Layer
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(APP_STATE.map);

  // Initialize Layer Groups in proper z-index order
  APP_STATE.layers.coverage = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.hazards = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.ghostRoutes = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.routes = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.facilities = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.incidents = L.layerGroup().addTo(APP_STATE.map);
  APP_STATE.layers.units = L.layerGroup().addTo(APP_STATE.map);

  // Map Click Listener for interactive modes
  APP_STATE.map.on('click', handleMapClick);
}

// ---------------------------------------------------------------------------
// UI Event Bindings
// ---------------------------------------------------------------------------
function bindUIControls() {
  // Top control bar buttons
  document.getElementById('btn-load-demo').addEventListener('click', () => apiPost('/api/scenario/load-demo'));
  document.getElementById('btn-replan').addEventListener('click', () => apiPost('/api/replan'));
  document.getElementById('btn-reset').addEventListener('click', () => apiPost('/api/scenario/reset'));

  // Toggle Coverage Layer
  const btnCov = document.getElementById('btn-toggle-coverage');
  btnCov.addEventListener('click', () => {
    APP_STATE.toggles.coverage = !APP_STATE.toggles.coverage;
    btnCov.dataset.active = APP_STATE.toggles.coverage;
    btnCov.innerText = `Coverage: ${APP_STATE.toggles.coverage ? 'ON' : 'OFF'}`;
    renderCoverage(APP_STATE.store.coverage);
  });

  // Toggle Ghost Layer
  const btnGhost = document.getElementById('btn-toggle-ghost');
  btnGhost.addEventListener('click', () => {
    APP_STATE.toggles.ghost = !APP_STATE.toggles.ghost;
    btnGhost.dataset.active = APP_STATE.toggles.ghost;
    btnGhost.innerText = `Ghost Plan: ${APP_STATE.toggles.ghost ? 'ON' : 'OFF'}`;
    renderGhostRoutes(APP_STATE.store.previous_plan);
  });

  // Mode Toggles
  const btnHazard = document.getElementById('btn-draw-hazard');
  const btnIncMap = document.getElementById('btn-add-incident-map');

  btnHazard.addEventListener('click', () => {
    toggleMode('draw_hazard', btnHazard);
  });

  btnIncMap.addEventListener('click', () => {
    toggleMode('add_incident', btnIncMap);
  });

  // Export SitRep
  document.getElementById('btn-export-sitrep').addEventListener('click', exportSitRep);

  // Text Intake Submission
  document.getElementById('intake-form').addEventListener('submit', handleTextIntake);

  // Confirmation Card Buttons
  document.getElementById('btn-commit-confirm').addEventListener('click', handleConfirmCommit);
  document.getElementById('btn-cancel-confirm').addEventListener('click', () => {
    document.getElementById('confirm-card').classList.add('hidden');
  });

  // Approval Modal Actions
  document.getElementById('btn-modal-approve').addEventListener('click', () => resolveApproval(true));
  document.getElementById('btn-modal-reject').addEventListener('click', () => resolveApproval(false));
}

function toggleMode(mode, triggerBtn) {
  if (APP_STATE.activeMode === mode) {
    APP_STATE.activeMode = null;
    triggerBtn.dataset.active = 'false';
    showToast(`Deactivated ${mode} mode.`);
  } else {
    // Reset any other mode button
    document.getElementById('btn-draw-hazard').dataset.active = 'false';
    document.getElementById('btn-add-incident-map').dataset.active = 'false';

    APP_STATE.activeMode = mode;
    triggerBtn.dataset.active = 'true';
    showToast(`Click anywhere on map to ${mode === 'draw_hazard' ? 'place hazard' : 'report incident'}.`);
  }
}

// ---------------------------------------------------------------------------
// Map Interaction: Add Incident / Hazard
// ---------------------------------------------------------------------------
function handleMapClick(e) {
  const { lat, lng } = e.latlng;

  if (APP_STATE.activeMode === 'add_incident') {
    openMapIncidentPopup(lat, lng);
    toggleMode('add_incident', document.getElementById('btn-add-incident-map'));
  } else if (APP_STATE.activeMode === 'draw_hazard') {
    const radiusStr = prompt('Enter hazard perimeter radius in meters:', '600');
    if (!radiusStr) return;
    const radius = parseFloat(radiusStr);
    const kind = prompt('Hazard kind (e.g., chemical_spill, flooding, road_block):', 'flooding') || 'flooding';

    apiPost('/api/hazards', {
      lat: roundCoord(lat),
      lng: roundCoord(lng),
      radius_m: radius,
      kind: kind
    });
    toggleMode('draw_hazard', document.getElementById('btn-draw-hazard'));
  }
}

function openMapIncidentPopup(lat, lng) {
  const popupContent = `
    <div style="font-family: inherit; font-size: 12px; color: #fff;">
      <h4 style="margin: 0 0 6px 0; color: #64b5f6;">Report Scene Incident</h4>
      <div style="margin-bottom: 6px;">
        <label>Type:</label>
        <select id="pop-inc-type" style="width:100%; background:#1b2430; color:#fff; border:1px solid #334; padding:3px;">
          <option value="cardiac_arrest">Cardiac Arrest</option>
          <option value="major_trauma" selected>Major Trauma</option>
          <option value="fire">Fire</option>
          <option value="building_collapse">Building Collapse</option>
          <option value="evacuation">Evacuation</option>
          <option value="minor_injury">Minor Injury</option>
        </select>
      </div>
      <div style="display:flex; gap:6px; margin-bottom:6px;">
        <div>
          <label>Severity (1-5):</label>
          <input type="number" id="pop-inc-sev" min="1" max="5" value="3" style="width:100%; background:#1b2430; color:#fff; border:1px solid #334; padding:3px;">
        </div>
        <div>
          <label>Affected:</label>
          <input type="number" id="pop-inc-people" min="0" value="2" style="width:100%; background:#1b2430; color:#fff; border:1px solid #334; padding:3px;">
        </div>
      </div>
      <div style="margin-bottom: 8px;">
        <label>Location Landmark:</label>
        <input type="text" id="pop-inc-loc" value="Map Coordinates (${lat.toFixed(4)}, ${lng.toFixed(4)})" style="width:100%; background:#1b2430; color:#fff; border:1px solid #334; padding:3px;">
      </div>
      <button id="pop-inc-submit" style="width:100%; background:#1976d2; color:#fff; border:none; padding:6px; border-radius:3px; font-weight:bold; cursor:pointer;">
        Dispatch Response
      </button>
    </div>
  `;

  const popup = L.popup()
    .setLatLng([lat, lng])
    .setContent(popupContent)
    .openOn(APP_STATE.map);

  setTimeout(() => {
    const submitBtn = document.getElementById('pop-inc-submit');
    if (submitBtn) {
      submitBtn.addEventListener('click', () => {
        const type = document.getElementById('pop-inc-type').value;
        const severity = parseInt(document.getElementById('pop-inc-sev').value, 10);
        const people = parseInt(document.getElementById('pop-inc-people').value, 10);
        const locText = document.getElementById('pop-inc-loc').value;

        apiPost('/api/incidents/manual', {
          type: type,
          severity: severity,
          people_affected: people,
          lat: roundCoord(lat),
          lng: roundCoord(lng),
          location_text: locText
        });
        APP_STATE.map.closePopup();
      });
    }
  }, 100);
}

// ---------------------------------------------------------------------------
// Multimodal Intake & Confirmation Card
// ---------------------------------------------------------------------------
async function handleTextIntake(e) {
  e.preventDefault();
  const inputEl = document.getElementById('intake-text');
  const text = inputEl.value.trim();
  if (!text) return;

  try {
    const res = await apiPost('/api/incidents/text', { text });
    inputEl.value = '';

    if (res.status === 'needs_confirmation') {
      displayConfirmationCard(res.extraction, 'text');
    } else {
      showToast('Incident processed and replanned.');
    }
  } catch (err) {
    showToast(`Intake error: ${err.message}`, 'error');
  }
}

function displayConfirmationCard(ext, source = 'text') {
  const card = document.getElementById('confirm-card');
  card.classList.remove('hidden');

  document.getElementById('confirm-confidence').innerText = `Confidence: ${(ext.confidence * 100).toFixed(0)}%`;
  document.getElementById('cf-type').value = ext.incident_type || 'major_trauma';
  document.getElementById('cf-severity').value = ext.severity || 3;
  document.getElementById('cf-people').value = ext.people_affected || 1;
  document.getElementById('cf-location').value = ext.location_text || '';
  document.getElementById('cf-lat').value = ext.lat ? ext.lat.toFixed(4) : '';
  document.getElementById('cf-lng').value = ext.lng ? ext.lng.toFixed(4) : '';
  card.dataset.source = source;
}

async function handleConfirmCommit() {
  const card = document.getElementById('confirm-card');
  const latVal = parseFloat(document.getElementById('cf-lat').value);
  const lngVal = parseFloat(document.getElementById('cf-lng').value);

  if (isNaN(latVal) || isNaN(lngVal)) {
    showToast('Valid latitude and longitude are required to dispatch.', 'error');
    return;
  }

  const payload = {
    type: document.getElementById('cf-type').value,
    severity: parseInt(document.getElementById('cf-severity').value, 10),
    people_affected: parseInt(document.getElementById('cf-people').value, 10),
    location_text: document.getElementById('cf-location').value,
    lat: roundCoord(latVal),
    lng: roundCoord(lngVal),
    source: card.dataset.source || 'text'
  };

  try {
    await apiPost('/api/incidents/confirm', payload);
    card.classList.add('hidden');
    showToast('Verified incident dispatched successfully.', 'success');
  } catch (err) {
    showToast(`Verification error: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// SSE Real-Time Stream
// ---------------------------------------------------------------------------
function initEventSource() {
  const evtSource = new EventSource('/api/stream');

  evtSource.onmessage = (e) => {
    try {
      const payload = JSON.parse(e.data);
      handleServerEvent(payload);
    } catch (err) {
      console.warn('Failed to parse SSE payload', err);
    }
  };

  evtSource.onerror = (err) => {
    console.warn('SSE stream reconnecting...', err);
  };
}

function handleServerEvent(evt) {
  if (evt.type === 'init') {
    updateFullState(evt.data);
  } else if (evt.type === 'event') {
    appendEventToFeed(evt.data);
  } else if (evt.type === 'plan_committed' || evt.type === 'incident_added' || evt.type === 'unit_updated' || evt.type === 'hazard_added' || evt.type === 'reset') {
    fetchInitialState();
  }
}

// ---------------------------------------------------------------------------
// Rendering & State Sync
// ---------------------------------------------------------------------------
async function fetchInitialState() {
  try {
    const state = await apiGet('/api/state');
    updateFullState(state);
  } catch (err) {
    showToast('Failed to fetch initial state', 'error');
  }
}

function updateFullState(state) {
  APP_STATE.store = state;

  renderUnits(state.units);
  renderIncidents(state.incidents);
  renderFacilities(state.facilities);
  renderHazards(state.hazards);
  renderActiveRoutes(state.current_plan, state.units, state.incidents);
  renderGhostRoutes(state.previous_plan);
  renderCoverage(state.coverage);
  renderStagingAdvice(state.staging_suggestions);
  renderReasoningCards(state.current_plan, state.pending_plan);
  renderUncoveredList(state.current_plan, state.incidents);

  // Sync event feed if provided in snapshot
  if (state.events && state.events.length > 0) {
    const feed = document.getElementById('event-feed');
    feed.innerHTML = '';
    state.events.forEach(appendEventToFeed);
  }
}

function renderUnits(units) {
  APP_STATE.layers.units.clearLayers();

  units.forEach(u => {
    const isDown = u.status === 'down';
    const isEnRoute = u.status === 'en_route';
    const emoji = UNIT_ICONS[u.type] || '🚑';

    const iconHtml = `
      <div style="font-size: 20px; line-height: 1; filter: ${isDown ? 'grayscale(100%) opacity(40%)' : 'none'};">
        ${emoji}
      </div>
    `;

    const icon = L.divIcon({
      html: iconHtml,
      className: 'unit-div-marker',
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    const marker = L.marker([u.lat, u.lng], { icon: icon });

    const popupHtml = `
      <div style="font-size: 12px; color: #fff;">
        <strong style="color: #64b5f6;">${u.name}</strong> (${u.id})<br>
        <strong>Type:</strong> ${u.type.replace('_', ' ')}<br>
        <strong>Status:</strong> <span style="text-transform:uppercase; color:${isDown ? '#ff5252' : isEnRoute ? '#ffb300' : '#00e676'}">${u.status}</span><br>
        ${u.assigned_incident_id ? `<strong>Target:</strong> ${u.assigned_incident_id}<br>` : ''}
        <div style="margin-top: 8px;">
          ${!isDown ? `<button onclick="setUnitStatus('${u.id}', 'down')" style="background:#b71c1c; color:#fff; border:none; padding:4px 8px; border-radius:3px; cursor:pointer;">Mark Unavailable</button>`
                    : `<button onclick="setUnitStatus('${u.id}', 'available')" style="background:#1b5e20; color:#fff; border:none; padding:4px 8px; border-radius:3px; cursor:pointer;">Restore Available</button>`}
        </div>
      </div>
    `;

    marker.bindPopup(popupHtml);
    APP_STATE.layers.units.addLayer(marker);
  });
}

function renderIncidents(incidents) {
  APP_STATE.layers.incidents.clearLayers();

  incidents.forEach(inc => {
    if (inc.status === 'resolved') return;

    let markerClass = 'marker-pulse-sev1';
    if (inc.severity === 3) markerClass = 'marker-pulse-sev3';
    else if (inc.severity >= 4) markerClass = 'marker-pulse-sev5';

    const icon = L.divIcon({
      html: `<div class="${markerClass}"></div>`,
      className: 'incident-div-marker',
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });

    const marker = L.marker([inc.lat, inc.lng], { icon: icon });

    const popupHtml = `
      <div style="font-size: 12px; color: #fff;">
        <strong style="color: #ff5252;">${inc.id}</strong> — ${inc.type.replace('_', ' ').toUpperCase()}<br>
        <strong>Severity:</strong> Level ${inc.severity} / 5<br>
        <strong>Casualties:</strong> ~${inc.people_affected} affected<br>
        <strong>Location:</strong> ${inc.location_text}<br>
        <strong>Status:</strong> <span style="font-weight:bold;">${inc.status.toUpperCase()}</span>
      </div>
    `;

    marker.bindPopup(popupHtml);
    APP_STATE.layers.incidents.addLayer(marker);
  });
}

function renderFacilities(facilities) {
  APP_STATE.layers.facilities.clearLayers();

  facilities.forEach(f => {
    const isHospital = f.kind === 'hospital';
    const emoji = isHospital ? '🏥' : '🏕️';
    const percentOccupied = Math.min(100, Math.round((f.occupied / f.capacity) * 100));

    const icon = L.divIcon({
      html: `<div style="font-size: 18px;">${emoji}</div>`,
      className: 'facility-div-marker',
      iconSize: [20, 20],
      iconAnchor: [10, 10]
    });

    const marker = L.marker([f.lat, f.lng], { icon: icon });

    const popupHtml = `
      <div style="font-size: 12px; color: #fff; width: 180px;">
        <strong style="color: #90caf9;">${f.name}</strong><br>
        <strong>Capacity:</strong> ${f.occupied} / ${f.capacity} beds (${percentOccupied}%)
        <div style="background: #263548; height: 6px; border-radius: 3px; margin-top: 4px; overflow: hidden;">
          <div style="background: ${percentOccupied > 80 ? '#ff1744' : '#00e676'}; width: ${percentOccupied}%; height: 100%;"></div>
        </div>
      </div>
    `;

    marker.bindPopup(popupHtml);
    APP_STATE.layers.facilities.addLayer(marker);
  });
}

function renderHazards(hazards) {
  APP_STATE.layers.hazards.clearLayers();

  hazards.forEach(h => {
    const circle = L.circle([h.lat, h.lng], {
      radius: h.radius_m,
      color: '#ff5722',
      fillColor: '#ff5722',
      fillOpacity: 0.25,
      weight: 2,
      dashArray: '4, 6'
    });

    circle.bindPopup(`
      <div style="font-size:12px; color:#fff;">
        <strong style="color:#ff7043;">HAZARD: ${h.kind.toUpperCase()}</strong> (${h.id})<br>
        <strong>Exclusion Radius:</strong> ${h.radius_m}m
      </div>
    `);
    APP_STATE.layers.hazards.addLayer(circle);
  });
}

function renderActiveRoutes(plan, units, incidents) {
  APP_STATE.layers.routes.clearLayers();
  if (!plan || !plan.assignments) return;

  const unitMap = Object.fromEntries(units.map(u => [u.id, u]));
  const incMap = Object.fromEntries(incidents.map(i => [i.id, i]));

  plan.assignments.forEach(assign => {
    const u = unitMap[assign.unit_id];
    const inc = incMap[assign.incident_id];
    if (!u || !inc) return;

    // Draw solid bold primary dispatch route
    const polyline = L.polyline([[u.lat, u.lng], [inc.lat, inc.lng]], {
      color: '#29b6f6',
      weight: 3.5,
      opacity: 0.85
    });

    polyline.bindTooltip(`${u.name} ➔ ${inc.id} (${assign.eta_min}m)`, {
      sticky: true,
      className: 'route-tooltip'
    });
    APP_STATE.layers.routes.addLayer(polyline);

    // Trigger unit animation if en_route
    if (u.status === 'en_route') {
      startUnitMovementAnimation(u, inc, assign.eta_min);
    }
  });
}

function renderGhostRoutes(previousPlan) {
  APP_STATE.layers.ghostRoutes.clearLayers();
  if (!APP_STATE.toggles.ghost || !previousPlan || !previousPlan.assignments) return;

  const unitMap = Object.fromEntries(APP_STATE.store.units.map(u => [u.id, u]));
  const incMap = Object.fromEntries(APP_STATE.store.incidents.map(i => [i.id, i]));

  previousPlan.assignments.forEach(assign => {
    const u = unitMap[assign.unit_id];
    const inc = incMap[assign.incident_id];
    if (!u || !inc) return;

    // Faint dashed ghost polyline representing previous assignment
    const ghostPoly = L.polyline([[u.lat, u.lng], [inc.lat, inc.lng]], {
      color: '#b388ff',
      weight: 2,
      opacity: 0.35,
      dashArray: '6, 8'
    });

    ghostPoly.bindTooltip(`Ghost: ${u.id} ➔ ${inc.id} (Prior Plan)`, { sticky: true });
    APP_STATE.layers.ghostRoutes.addLayer(ghostPoly);
  });
}

function renderCoverage(coverage) {
  APP_STATE.layers.coverage.clearLayers();
  if (!APP_STATE.toggles.coverage || !coverage || !coverage.cells) return;

  const cellDelta = 0.007; // Approximate half-width for grid box rendering

  coverage.cells.forEach(cell => {
    const color = cell.covered ? '#00e676' : (cell.nearest_eta < 20 ? '#ffb300' : '#ff1744');
    const opacity = cell.covered ? 0.05 : 0.18;

    const bounds = [
      [cell.lat - cellDelta, cell.lng - cellDelta],
      [cell.lat + cellDelta, cell.lng + cellDelta]
    ];

    const rect = L.rectangle(bounds, {
      color: color,
      weight: 1,
      opacity: 0.2,
      fillColor: color,
      fillOpacity: opacity
    });

    rect.bindTooltip(`Nearest ETA: ${cell.nearest_eta}m ${cell.covered ? '(Covered)' : '(Deficit)'}`, { sticky: true });
    APP_STATE.layers.coverage.addLayer(rect);
  });
}

// ---------------------------------------------------------------------------
// Unit Movement Simulation & Arrival Dispatch
// ---------------------------------------------------------------------------
function startUnitMovementAnimation(unit, incident, etaMinutes) {
  // Prevent duplicate animation threads
  if (APP_STATE.animations[unit.id]) return;

  // Duration in ms: (eta_min / SIM_SPEED) * 60 * 1000
  const durationMs = Math.max(3000, (etaMinutes / APP_STATE.simSpeed) * 60 * 1000);
  const startTime = performance.now();
  const startLat = unit.lat;
  const startLng = unit.lng;
  const targetLat = incident.lat;
  const targetLng = incident.lng;

  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(1.0, elapsed / durationMs);

    // Linear interpolation of coordinates
    unit.lat = startLat + (targetLat - startLat) * progress;
    unit.lng = startLng + (targetLng - startLng) * progress;

    if (progress < 1.0) {
      APP_STATE.animations[unit.id] = requestAnimationFrame(step);
    } else {
      // Arrival complete
      delete APP_STATE.animations[unit.id];
      apiPost(`/api/units/${unit.id}/arrived`).then(() => {
        showToast(`${unit.name} has ARRIVED at ${incident.id}.`);
      });
    }
  }

  APP_STATE.animations[unit.id] = requestAnimationFrame(step);
}

// ---------------------------------------------------------------------------
// Right Panel: Reasoning Cards, Uncovered Gaps & Agent Feed
// ---------------------------------------------------------------------------
function renderReasoningCards(currentPlan, pendingPlan) {
  const container = document.getElementById('reasoning-cards');
  const badge = document.getElementById('plan-version-badge');
  const plan = currentPlan;

  if (plan) {
    badge.innerText = `v${plan.version}`;
  }

  container.innerHTML = '';

  if (!plan || !plan.assignments || plan.assignments.length === 0) {
    container.innerHTML = '<div class="empty-state">No active unit assignments.</div>';
    return;
  }

  plan.assignments.forEach(assign => {
    const cardEl = document.createElement('div');
    cardEl.className = 'reasoning-card';

    cardEl.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
        <strong style="color:#64b5f6;">${assign.unit_id} ➔ ${assign.incident_id}</strong>
        <span class="pill">ETA ${assign.eta_min}m</span>
      </div>
      <div class="reasoning-text">Assigned to mitigate expected harm (${(assign.harm * 100).toFixed(1)}%).</div>
      <div class="metric-pills">
        <span class="pill">Harm: ${assign.harm.toFixed(3)}</span>
      </div>
    `;
    container.appendChild(cardEl);
  });
}

function renderUncoveredList(plan, incidents) {
  const container = document.getElementById('uncovered-list');
  const countBadge = document.getElementById('deficit-count');

  if (!plan || !plan.uncovered || plan.uncovered.length === 0) {
    countBadge.innerText = '0';
    countBadge.className = 'badge';
    container.innerHTML = '<div class="empty-state-success">All incident demands fulfilled.</div>';
    return;
  }

  let totalMissing = 0;
  container.innerHTML = '';

  plan.uncovered.forEach(unc => {
    totalMissing += unc.missing;
    const inc = incidents.find(i => i.id === unc.incident_id);
    const sev = inc ? inc.severity : 3;

    const div = document.createElement('div');
    div.className = 'uncovered-item';
    div.innerHTML = `
      <div>
        <div class="uncovered-title">${unc.incident_id} (Severity ${sev})</div>
        <div style="font-size:11px; color:#fff;">Missing: <strong>${unc.missing}x ${unc.unit_type.replace('_', ' ')}</strong></div>
      </div>
      <span class="badge badge-danger">DEFICIT</span>
    `;
    container.appendChild(div);
  });

  countBadge.innerText = `${totalMissing}`;
  countBadge.className = 'badge badge-danger';
}

function renderStagingAdvice(suggestions) {
  const box = document.getElementById('staging-box');
  const content = document.getElementById('staging-content');

  if (!suggestions || suggestions.length === 0) {
    box.classList.add('hidden');
    return;
  }

  box.classList.remove('hidden');
  const item = suggestions[0];
  content.innerHTML = `
    <div style="font-size:11px; color:#fff;">
      <strong>${item.unit_name} (${item.unit_id})</strong><br>
      ${item.reason}
      <div style="margin-top:6px;">
        <span class="pill">Target: ${item.target_lat}, ${item.target_lng}</span>
      </div>
    </div>
  `;
}

function appendEventToFeed(evt) {
  const feed = document.getElementById('event-feed');
  const item = document.createElement('div');
  item.className = 'event-item';

  const timeStr = evt.ts ? evt.ts.split('T')[1].slice(0, 8) : new Date().toLocaleTimeString();

  item.innerHTML = `
    <span class="event-time">${timeStr}</span>
    <span class="event-agent">${evt.agent}</span>
    <span>${evt.message}</span>
  `;

  feed.prepend(item);

  // Keep feed clamped to 50 items
  while (feed.children.length > 50) {
    feed.removeChild(feed.lastChild);
  }
}

// ---------------------------------------------------------------------------
// Human-in-the-Loop Approval Modal
// ---------------------------------------------------------------------------
function checkAndDisplayApproval(replanResult) {
  if (!replanResult || !replanResult.interrupted) return;

  APP_STATE.pendingThreadId = replanResult.thread_id;
  const modal = document.getElementById('approval-modal');
  const flagsList = document.getElementById('modal-flags-list');
  const diffBox = document.getElementById('modal-diff-summary');

  flagsList.innerHTML = '';
  diffBox.innerHTML = '';

  // Render risk flags
  (replanResult.flags || []).forEach(f => {
    const p = document.createElement('p');
    p.style.margin = '3px 0';
    p.innerHTML = `<strong>[FLAG]</strong> ${f.reason}`;
    flagsList.appendChild(p);
  });

  // Render reasoning cards inside modal
  (replanResult.cards || []).forEach(c => {
    const cardEl = document.createElement('div');
    cardEl.style.cssText = 'padding:6px; background:#1b2430; border-radius:3px; margin-bottom:4px; font-size:12px;';
    cardEl.innerHTML = `
      <strong>${c.unit_id} Diverted:</strong> ${c.text}<br>
      <span style="color:#ffb300;">ETA Change: ${c.numbers.eta_before}m ➔ ${c.numbers.eta_after}m</span>
    `;
    diffBox.appendChild(cardEl);
  });

  modal.classList.remove('hidden');
}

async function resolveApproval(approved) {
  const threadId = APP_STATE.pendingThreadId;
  if (!threadId) return;

  try {
    await apiPost(`/api/approval/${threadId}`, { approve: approved });
    document.getElementById('approval-modal').classList.add('hidden');
    APP_STATE.pendingThreadId = null;
    showToast(`Plan ${approved ? 'APPROVED & COMMITTED' : 'REJECTED: Applied safe non-diverting plan'}.`, approved ? 'success' : 'error');
    fetchInitialState();
  } catch (err) {
    showToast(`Approval error: ${err.message}`, 'error');
  }
}

// ---------------------------------------------------------------------------
// SitRep Export & Status Helpers
// ---------------------------------------------------------------------------
async function exportSitRep() {
  try {
    const res = await fetch('/api/report/methane');
    const markdown = await res.text();

    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SitRep_METHANE_${new Date().toISOString().slice(0, 19).replace(/[:]/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('METHANE SitRep exported successfully.', 'success');
  } catch (err) {
    showToast(`Failed to export SitRep: ${err.message}`, 'error');
  }
}

window.setUnitStatus = async function(unitId, status) {
  try {
    await apiPost(`/api/resources/${unitId}/status`, { status });
    showToast(`Unit ${unitId} updated to ${status}.`);
    fetchInitialState();
  } catch (err) {
    showToast(`Failed to set unit status: ${err.message}`, 'error');
  }
};

// ---------------------------------------------------------------------------
// Network Helpers & Toasts
// ---------------------------------------------------------------------------
async function apiGet(endpoint) {
  const resp = await fetch(endpoint);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return await resp.json();
}

async function apiPost(endpoint, body = {}) {
  const resp = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    const errorMsg = err.detail || 'Server error';
    showToast(errorMsg, 'error');
    throw new Error(errorMsg);
  }

  const result = await resp.json();
  if (result.plan_result) {
    checkAndDisplayApproval(result.plan_result);
  }
  return result;
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;
  toast.innerText = message;
  container.appendChild(toast);

  setTimeout(() => {
    if (toast.parentNode) {
      container.removeChild(toast);
    }
  }, 4000);
}

function roundCoord(val) {
  return Math.round(val * 10000) / 10000;
}
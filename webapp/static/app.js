const $ = id => document.getElementById(id);

let vehicles = [];
let network = null;
let selectedVehicle = null;
let selectedPriority = "balanced";
let routeData = null;
let mapBounds = null;
let nodeLookup = new Map();
let animationTimer = null;
let driveState = null;
let clockTimer = null;

const DEMO_VEHICLES = ["V000001", "V000011"];

async function get(url, opts) {
  const response = await fetch(url, opts);

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json();
}

function fmt(n, d = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) {
    return "—";
  }

  return Number(n).toFixed(d);
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}


/* =========================================================
   INITIAL BOOT
========================================================= */

async function boot() {
  setTimeout(() => {
    const splash = $("splash");

    if (splash) {
      splash.style.opacity = "0";

      setTimeout(() => {
        splash.remove();
      }, 500);
    }
  }, 900);

  try {
    vehicles = await get("/api/vehicles");

    const demo = DEMO_VEHICLES
      .map(id =>
        vehicles.find(v => String(v.vehicle_id) === id)
      )
      .filter(Boolean);

    const choices = demo.length
      ? demo
      : vehicles.slice(0, 2);

    $("vehicleCards").innerHTML = choices.map((v, i) => `
      <button
        class="vehicle-choice ${i === 0 ? "selected" : ""}"
        data-id="${v.vehicle_id}"
      >
        <span class="vehicle-choice-icon">
          ${(v.vehicle_type || "CAR").slice(0, 2).toUpperCase()}
        </span>

        <span class="vehicle-choice-main">
          <b>${v.vehicle_id}</b>
          <small>
            ${v.vehicle_type || "Vehicle"} ·
            ${v.fuel_type || "—"}
          </small>
        </span>

        <span class="vehicle-choice-check">✓</span>
      </button>
    `).join("");

    selectedVehicle = choices[0]?.vehicle_id || null;

    document.querySelectorAll(".vehicle-choice").forEach(btn => {
      btn.onclick = () => {

        document
          .querySelectorAll(".vehicle-choice")
          .forEach(x => x.classList.remove("selected"));

        btn.classList.add("selected");

        selectedVehicle = btn.dataset.id;

        $("enterBtn").disabled = false;
      };
    });

    $("enterBtn").disabled = !selectedVehicle;

    $("enterBtn").onclick = enterVehicle;

  } catch (error) {

    $("vehicleCards").innerHTML = `
      <div class="error-box">
        Unable to load vehicles: ${error.message}
      </div>
    `;
  }
}


/* =========================================================
   ENTER VEHICLE
========================================================= */

async function enterVehicle() {

  if (!selectedVehicle) {
    return;
  }

  $("login").classList.add("leaving");

  setTimeout(async () => {

    $("login").classList.add("hidden");

    $("app").classList.remove("hidden");

    await loadDashboard();

  }, 280);
}


/* =========================================================
   LOAD DASHBOARD
========================================================= */

async function loadDashboard() {

  try {

    const [
      summary,
      net,
      vehicle,
      analytics,
      destinations
    ] = await Promise.all([

      get("/api/summary"),

      get("/api/network"),

      get(
        `/api/vehicle/${encodeURIComponent(selectedVehicle)}`
      ),

      get("/api/analytics"),

      get(
        `/api/destinations?vehicle_id=${encodeURIComponent(selectedVehicle)}`
      )

    ]);

    network = net;

    $("datasetVehicles").textContent =
      Number(summary.vehicles).toLocaleString();

    $("datasetEvents").textContent =
      Number(
        summary.telemetry_events ||
        summary.telemetry
      ).toLocaleString();

    renderVehicle(vehicle);

    renderAnalytics(analytics);

    renderMap();

    renderDestinations(destinations);

    $("routePanel").classList.add("hidden");

    $("setupSection").classList.remove("hidden");

    $("driveInfo").classList.add("hidden");

    $("driveOverlay").classList.add("hidden");

    $("headline").textContent =
      "Where are you going?";

    $("subline").textContent =
      "Choose a real road-network destination and driving preference.";

    $("mapBadge").textContent =
      "Select a destination to begin";

    $("mapTitle").textContent =
      "ROAD NETWORK";

    $("clock").textContent =
      new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
      });

    clearInterval(clockTimer);

    clockTimer = setInterval(() => {

      $("clock").textContent =
        new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit"
        });

    }, 30000);

  } catch (error) {

    console.error("Dashboard loading error:", error);

    $("mapBadge").textContent =
      "Unable to load dashboard";

  }
}


/* =========================================================
   DESTINATIONS
========================================================= */

function renderDestinations(destinations) {

  $("destination").innerHTML =
    `<option value="">Choose a destination</option>` +

    destinations.map(d => `
      <option value="${d.id}">
        ${d.name} · ${d.id}
      </option>
    `).join("");
}


/* =========================================================
   VEHICLE DATA
========================================================= */

function renderVehicle(data) {

  const profile = data.profile || {};

  const live =
    data.live ||
    data.latest_telemetry ||
    {};

  $("vehicleName").textContent =
    profile.vehicle_id || selectedVehicle;

  $("vehicleMeta").textContent =
    `${profile.vehicle_type || "Vehicle"} ·
     ${profile.fuel_type || "—"} ·
     ${profile.driving_style || "—"}`;

  setTelemetry(
    live,
    data.location
  );
}


/* =========================================================
   TELEMETRY
========================================================= */

function setTelemetry(live, location) {

  const speed =
    Number(live.speed_kmh || 0);

  const fuel =
    Number(live.fuel_level_pct || 0);

  const traffic =
    Number(live.traffic_level || 0) * 100;

  $("speed").textContent =
    fmt(speed, 0);

  $("sideSpeed").textContent =
    `${fmt(speed, 0)} km/h`;

  $("fuel").textContent =
    `${fmt(fuel, 0)}%`;

  $("sideFuel").textContent =
    `${fmt(fuel, 0)}%`;

  $("traffic").textContent =
    `${fmt(traffic, 0)}%`;

  $("trafficBar").style.width =
    `${clamp(traffic, 2, 100)}%`;

  $("state").textContent =
    live.vehicle_state || "MOVING";

  $("segment").textContent =
    live.segment_id || "—";

  $("fuelBar").style.width =
    `${clamp(fuel, 2, 100)}%`;

  $("location").textContent =
    location
      ? `${fmt(location.lat, 4)}, ${fmt(location.lon, 4)}`
      : "—";
}


/* =========================================================
   REAL ROAD NETWORK MAP
========================================================= */

function renderMap() {

  const width = 1000;
  const height = 600;

  const points =
    network.nodes || [];

  if (!points.length) {
    return;
  }

  const lats =
    points.map(n => Number(n.lat));

  const lons =
    points.map(n => Number(n.lon));

  mapBounds = {

    minLat: Math.min(...lats),

    maxLat: Math.max(...lats),

    minLon: Math.min(...lons),

    maxLon: Math.max(...lons)

  };

  const latRange =
    Math.max(
      0.000001,
      mapBounds.maxLat -
      mapBounds.minLat
    );

  const lonRange =
    Math.max(
      0.000001,
      mapBounds.maxLon -
      mapBounds.minLon
    );

  const xy = node => ({

    x:
      35 +
      (
        (Number(node.lon) -
          mapBounds.minLon) /
        lonRange
      ) * 930,

    y:
      565 -
      (
        (Number(node.lat) -
          mapBounds.minLat) /
        latRange
      ) * 530

  });

  nodeLookup =
    new Map(
      points.map(
        n => [String(n.id), n]
      )
    );

  window.smartRouteXY = xy;

  $("roads").innerHTML =
    (network.segments || [])
      .map(segment => {

        const a =
          nodeLookup.get(
            String(segment.from)
          );

        const b =
          nodeLookup.get(
            String(segment.to)
          );

        if (!a || !b) {
          return "";
        }

        const p = xy(a);
        const q = xy(b);

        return `
          <line
            x1="${p.x}"
            y1="${p.y}"
            x2="${q.x}"
            y2="${q.y}"
          />
        `;

      })
      .join("");
}


/* =========================================================
   CLEAR ROUTE
========================================================= */

function clearRoute() {

  $("routeLayer").innerHTML = "";

  $("markers").innerHTML = "";

  $("car").classList.add("hidden");
}


/* =========================================================
   DRAW EXACT DIJKSTRA ROUTE
========================================================= */

function drawRoute(recommended) {

  clearRoute();

  const path =
    recommended.nodes || [];

  if (path.length < 2) {
    return;
  }

  const points = [];

  for (let i = 0; i < path.length; i++) {

    const node =
      nodeLookup.get(
        String(path[i])
      );

    if (!node) {
      continue;
    }

    points.push(
      window.smartRouteXY(node)
    );
  }

  if (points.length < 2) {
    return;
  }

  const polylinePoints =
    points
      .map(p => `${p.x},${p.y}`)
      .join(" ");

  $("routeLayer").innerHTML = `

    <polyline
      class="route-glow"
      points="${polylinePoints}"
    />

    <polyline
      class="route-line"
      points="${polylinePoints}"
    />

  `;

  const start =
    points[0];

  const end =
    points[points.length - 1];

  $("markers").innerHTML = `

    <circle
      class="marker-ring"
      cx="${start.x}"
      cy="${start.y}"
      r="12"
    />

    <circle
      class="marker"
      cx="${start.x}"
      cy="${start.y}"
      r="6"
    />

    <circle
      class="destination-ring"
      cx="${end.x}"
      cy="${end.y}"
      r="13"
    />

    <circle
      class="destination-marker"
      cx="${end.x}"
      cy="${end.y}"
      r="6"
    />

  `;

  $("car").setAttribute(
    "cx",
    start.x
  );

  $("car").setAttribute(
    "cy",
    start.y
  );

  $("car").classList.remove("hidden");
}


/* =========================================================
   ROUTE POINTS
========================================================= */

function buildRoutePoints(path) {

  return path

    .map(id =>
      nodeLookup.get(
        String(id)
      )
    )

    .filter(Boolean)

    .map(window.smartRouteXY);
}


/* =========================================================
   INTERPOLATION
========================================================= */

function interpolate(a, b, t) {

  return {

    x:
      a.x +
      (b.x - a.x) * t,

    y:
      a.y +
      (b.y - a.y) * t

  };
}


/* =========================================================
   STOP ANIMATION
========================================================= */

function stopDriveAnimation() {

  if (animationTimer) {

    cancelAnimationFrame(
      animationTimer
    );

    animationTimer = null;
  }
}


/* =========================================================
   START DRIVE
========================================================= */

function startDrive() {

  if (
    !routeData ||
    !routeData.recommended ||
    !routeData.recommended.nodes ||
    routeData.recommended.nodes.length < 2
  ) {
    return;
  }

  stopDriveAnimation();

  const points =
    buildRoutePoints(
      routeData.recommended.nodes
    );

  if (points.length < 2) {
    return;
  }

  const totalKm =
    Number(
      routeData.recommended.distance_km || 1
    );

  const totalMin =
    Number(
      routeData.recommended.time_min || 1
    );

  /*
   * Presentation duration.
   *
   * The actual route geometry is used.
   * Only the playback speed is accelerated
   * for the demo.
   */

  const durationMs =
    clamp(
      totalMin * 900,
      12000,
      30000
    );

  const started =
    performance.now();

  driveState = {

    points,

    totalKm,

    totalMin,

    started,

    durationMs

  };

  $("setupSection")
    .classList
    .add("hidden");

  $("driveInfo")
    .classList
    .remove("hidden");

  $("driveOverlay")
    .classList
    .remove("hidden");

  $("routePanel")
    .classList
    .add("hidden");

  $("headerStatus").textContent =
    "NAVIGATING";

  $("mapTitle").textContent =
    "LIVE SMARTROUTE";

  $("mapBadge").textContent =
    "Following the calculated road path";

  $("headline").textContent =
    "Navigating to destination";

  $("subline").textContent =
    "Vehicle position follows the exact route returned by the routing engine.";

  $("overlayDestination").textContent =
    $("destination")
      .selectedOptions[0]
      ?.textContent ||
    "Destination";

  $("startDriveBtn")
    .classList
    .add("hidden");


  /* -----------------------------------------
     ACTUAL ROUTE ANIMATION
  ----------------------------------------- */

  const step = now => {

    if (!driveState) {
      return;
    }

    const elapsed =
      now -
      driveState.started;

    const progress =
      clamp(
        elapsed /
          driveState.durationMs,
        0,
        1
      );

    /*
     * Map progress onto the ACTUAL
     * Dijkstra route nodes.
     */

    const scaled =
      progress *
      (driveState.points.length - 1);

    const index =
      Math.min(
        driveState.points.length - 2,
        Math.floor(scaled)
      );

    const localT =
      scaled -
      index;

    const position =
      interpolate(
        driveState.points[index],
        driveState.points[index + 1],
        localT
      );

    $("car").setAttribute(
      "cx",
      position.x
    );

    $("car").setAttribute(
      "cy",
      position.y
    );


    /* -----------------------------------------
       TRIP METRICS
    ----------------------------------------- */

    const remainingKm =
      driveState.totalKm *
      (1 - progress);

    const remainingMin =
      driveState.totalMin *
      (1 - progress);

    $("overlayRemaining").textContent =
      `${remainingKm.toFixed(1)} km`;

    $("overlayEta").textContent =
      `${Math.max(
        0,
        Math.ceil(remainingMin)
      )} min`;

    $("sideRemaining").textContent =
      `${remainingKm.toFixed(1)} km`;

    $("sideEta").textContent =
      `${Math.max(
        0,
        Math.ceil(remainingMin)
      )} min`;


    /* -----------------------------------------
       DYNAMIC SPEED
    ----------------------------------------- */

    const baseSpeed =
      Math.max(
        18,
        Math.min(
          75,
          driveState.totalKm /
          Math.max(
            driveState.totalMin,
            0.1
          ) *
          60
        )
      );

    const wave =
      Math.sin(
        progress *
        Math.PI *
        6
      ) * 7;

    const speed =
      clamp(
        baseSpeed + wave,
        12,
        85
      );

    $("speed").textContent =
      Math.round(speed);

    $("sideSpeed").textContent =
      `${Math.round(speed)} km/h`;


    /* -----------------------------------------
       FUEL
    ----------------------------------------- */

    const startFuel =
      Number(
        routeData.vehicleFuelStart || 55
      );

    const fuelUsed =
      Number(
        routeData.recommended.fuel_l || 0
      );

    const currentFuel =
      Math.max(
        5,
        startFuel -
        fuelUsed * progress
      );

    $("fuel").textContent =
      `${currentFuel.toFixed(0)}%`;

    $("sideFuel").textContent =
      `${currentFuel.toFixed(0)}%`;

    $("fuelBar").style.width =
      `${clamp(
        currentFuel,
        2,
        100
      )}%`;


    /* -----------------------------------------
       CONTINUE OR ARRIVE
    ----------------------------------------- */

    if (progress < 1) {

      animationTimer =
        requestAnimationFrame(step);

    } else {

      finishDrive();

    }
  };

  animationTimer =
    requestAnimationFrame(step);
}


/* =========================================================
   FINISH DRIVE
========================================================= */

function finishDrive() {

  stopDriveAnimation();

  const path =
    routeData.recommended.nodes;

  const endNode =
    nodeLookup.get(
      String(
        path[path.length - 1]
      )
    );

  if (endNode) {

    const end =
      window.smartRouteXY(
        endNode
      );

    $("car").setAttribute(
      "cx",
      end.x
    );

    $("car").setAttribute(
      "cy",
      end.y
    );
  }

  $("headerStatus").textContent =
    "ARRIVED";

  $("mapBadge").textContent =
    "Arrived at destination";

  $("headline").textContent =
    "Trip complete";

  $("subline").textContent =
    `${routeData.recommended.distance_km} km · ` +
    `${routeData.recommended.time_min} min · ` +
    `${routeData.recommended.fuel_l} L`;

  $("sideRemaining").textContent =
    "0.0 km";

  $("sideEta").textContent =
    "Arrived";

  $("state").textContent =
    "ARRIVED";

  $("rerouteBtn").disabled = true;
}


/* =========================================================
   FIND ROUTE
========================================================= */

async function findRoute(hourOverride = null) {

  const destination =
    $("destination").value;

  if (!destination) {

    $("mapBadge").textContent =
      "Choose a destination first";

    return;
  }

  stopDriveAnimation();

  try {

    const vehicle =
      await get(
        `/api/vehicle/${encodeURIComponent(
          selectedVehicle
        )}`
      );

    const source =
      vehicle.current_node;

    const hour =
      hourOverride === null
        ? Number($("hour").value)
        : Number(hourOverride);

    $("mapBadge").textContent =
      "Calculating from the real road network…";


    const result =
      await get(
        `/api/route` +
        `?vehicle_id=${encodeURIComponent(selectedVehicle)}` +
        `&source=${encodeURIComponent(source)}` +
        `&destination=${encodeURIComponent(destination)}` +
        `&priority=${encodeURIComponent(selectedPriority)}` +
        `&hour=${hour}`
      );

    routeData =
      result;

    routeData.vehicleFuelStart =
      Number(
        vehicle.live?.fuel_level_pct ||
        55
      );

    drawRoute(
      result.recommended
    );

    renderRoute(
      result
    );

    $("headline").textContent =
      "Route ready.";

    $("subline").textContent =
      `${selectedVehicle} · ` +
      `${$("destination").selectedOptions[0]?.textContent || destination}`;

    $("mapBadge").textContent =
      `${result.recommended.distance_km} km · ` +
      `${result.recommended.time_min} min · ` +
      `${result.recommended.traffic_pct}% traffic`;

    $("startDriveBtn")
      .classList
      .remove("hidden");

    $("routePanel")
      .classList
      .remove("hidden");

  } catch (error) {

    console.error(
      "Route calculation error:",
      error
    );

    $("mapBadge").textContent =
      "Route unavailable — choose another destination";
  }
}


/* =========================================================
   ROUTE RESULT UI
========================================================= */

function renderRoute(result) {

  const route =
    result.recommended;

  $("routeDistance").textContent =
    route.distance_km;

  $("routeTime").textContent =
    route.time_min;

  $("routeFuel").textContent =
    route.fuel_l;

  $("routeTraffic").textContent =
    `${route.traffic_pct}%`;

  $("routeTitle").textContent =
    `${route.distance_km} km smart route`;

  $("explanation").innerHTML =
    (result.explanation || [])
      .map(
        text =>
          `<div>✓ ${text}</div>`
      )
      .join("");

  $("whyText").innerHTML =
    (result.explanation || [])
      .map(
        text =>
          `<div>• ${text}</div>`
      )
      .join("");
}


/* =========================================================
   ANALYTICS
========================================================= */

function renderAnalytics(data) {

  const values =
    data.hourly_traffic || [];

  const max =
    Math.max(
      ...values,
      0.01
    );

  $("trafficChart").innerHTML =
    values.map(
      (value, hour) => `
        <div
          class="bar ${hour === 18 ? "hot" : ""}"
          style="
            height:
            ${Math.max(
              3,
              value / max * 95
            )}%
          "
          title="
            ${hour}:00 ·
            ${(value * 100).toFixed(0)}%
          "
        ></div>
      `
    ).join("");


  const items =
    Object.entries(
      data.fleet_mix || {}
    ).slice(0, 5);

  const maxFleet =
    Math.max(
      ...items.map(
        x => x[1]
      ),
      1
    );

  $("fleetChart").innerHTML =
    items.map(
      ([type, count]) => `
        <div class="fleet-row">

          <label>${type}</label>

          <i>
            <b
              style="
                width:
                ${count / maxFleet * 100}%
              "
            ></b>
          </i>

          <span>${count}</span>

        </div>
      `
    ).join("");
}


/* =========================================================
   DRIVING MODE BUTTONS
========================================================= */

document
  .querySelectorAll(".mode")
  .forEach(button => {

    button.onclick = () => {

      document
        .querySelectorAll(".mode")
        .forEach(
          x =>
            x.classList.remove("active")
        );

      button.classList.add("active");

      selectedPriority =
        button.dataset.priority;

    };

  });


/* =========================================================
   DESTINATION CHANGE
========================================================= */

$("destination").onchange = () => {

  if (!$("destination").value) {

    $("routePanel")
      .classList
      .add("hidden");

    clearRoute();

    $("startDriveBtn")
      .classList
      .add("hidden");

  } else {

    $("startDriveBtn")
      .classList
      .add("hidden");

    $("mapBadge").textContent =
      "Destination selected · calculate SmartRoute";
  }
};


/* =========================================================
   HOUR SLIDER
========================================================= */

$("hour").oninput = event => {

  $("hourLabel").textContent =
    `${String(
      event.target.value
    ).padStart(2, "0")}:00`;

};


/* =========================================================
   ROUTE BUTTON
========================================================= */

$("routeBtn").onclick = () => {

  findRoute();

};


/* =========================================================
   START DRIVE BUTTON
========================================================= */

$("startDriveBtn").onclick = () => {

  startDrive();

};


/* =========================================================
   REROUTE
========================================================= */

$("rerouteBtn").onclick = async () => {

  if (!routeData) {
    return;
  }

  const nextHour =
    Math.min(
      23,
      Number($("hour").value) + 2
    );

  $("hour").value =
    nextHour;

  $("hourLabel").textContent =
    `${String(
      nextHour
    ).padStart(2, "0")}:00`;

  $("rerouteBtn").disabled =
    true;

  await findRoute(
    nextHour
  );

  $("rerouteBtn").disabled =
    false;
};


/* =========================================================
   END TRIP
========================================================= */

$("endTripBtn").onclick = () => {

  stopDriveAnimation();

  routeData = null;

  driveState = null;

  $("app")
    .classList
    .add("hidden");

  $("login")
    .classList
    .remove("hidden");

  $("login")
    .classList
    .remove("leaving");

  $("headerStatus").textContent =
    "VEHICLE ONLINE";

  clearRoute();

};


/* =========================================================
   START APPLICATION
========================================================= */

boot();
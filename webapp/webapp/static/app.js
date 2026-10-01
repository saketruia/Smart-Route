const $ = id => document.getElementById(id);
let vehicles = [], nodes = [], network = null, selectedVehicle = null, selectedPriority = "balanced";
let routeData = null, mapBounds = null, animationTimer = null, rerouteHour = 18;

async function get(url, opts){ const r = await fetch(url, opts); if(!r.ok) throw new Error(await r.text()); return r.json(); }

function fmt(n,d=1){ return n==null ? "—" : Number(n).toFixed(d); }

async function boot(){
  setTimeout(()=>{$("splash").style.opacity="0";setTimeout(()=>$("splash").remove(),500)},1200);
  try{
    vehicles=await get("/api/vehicles");
    $("vehicleSelect").innerHTML=vehicles.map(v=>`<option value="${v.vehicle_id}">${v.vehicle_id} · ${v.vehicle_type||"Vehicle"} · ${v.fuel_type||""}</option>`).join("");
    updateVehiclePreview();
    $("vehicleSelect").onchange=updateVehiclePreview;
    $("enterBtn").onclick=enterVehicle;
  }catch(e){$("vehiclePreview").textContent="Unable to load vehicle dataset: "+e.message}
}
function updateVehiclePreview(){
  const v=vehicles.find(x=>String(x.vehicle_id)==$("vehicleSelect").value)||vehicles[0];
  if(!v)return;
  $("vehiclePreview").innerHTML=`<b>${v.vehicle_type||"Connected vehicle"}</b> · ${v.fuel_type||"—"} · ${v.rated_efficiency_kmpl||"—"} km/L · ${v.driving_style||"—"} driving profile`;
}
async function enterVehicle(){
  selectedVehicle=$("vehicleSelect").value;
  $("login").classList.add("hidden"); $("app").classList.remove("hidden");
  await loadDashboard();
}
async function loadDashboard(){
  const [summary,net,vs,analytics]=await Promise.all([
    get("/api/summary"),get("/api/network"),get("/api/vehicle/"+encodeURIComponent(selectedVehicle)),get("/api/analytics")
  ]);
  network=net; nodes=await get("/api/nodes");
  $("datasetVehicles").textContent=summary.vehicles.toLocaleString();
  $("datasetEvents").textContent=summary.telemetry_events.toLocaleString();
  renderVehicle(vs); renderAnalytics(analytics); renderMap();
  $("destination").innerHTML=nodes.map(n=>`<option value="${n.id}">${n.id}</option>`).join("");
  const destination=nodes[Math.min(120,nodes.length-1)]?.id;
  $("destination").value=destination||"";
  $("clock").textContent=new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
  setInterval(()=>{$("clock").textContent=new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})},30000);
  await findRoute();
}
function renderVehicle(v){
  const p=v.profile||{}, l=v.live||{};
  $("vehicleName").textContent=v.profile.vehicle_id||selectedVehicle;
  $("vehicleMeta").textContent=`${p.vehicle_type||"Vehicle"} · ${p.fuel_type||"—"} · ${p.driving_style||"—"}`;
  setTelemetry(l,v.location);
}
function setTelemetry(l,loc){
  const speed=Number(l.speed_kmh||0), fuel=Number(l.fuel_level_pct||0), traffic=Number(l.traffic_level||0)*100;
  $("speed").textContent=fmt(speed,0);$("sideSpeed").textContent=fmt(speed,0);
  $("fuel").textContent=fmt(fuel,0)+"%";$("sideFuel").textContent=fmt(fuel,0);
  $("traffic").textContent=fmt(traffic,0)+"%";$("sideTraffic").textContent=traffic<45?"LOW":traffic<70?"MOD":"HIGH";
  $("state").textContent=l.vehicle_state||"MOVING";$("sideState").textContent=l.vehicle_state||"MOVING";
  $("segment").textContent=l.segment_id||"—";
  $("fuelBar").style.width=Math.max(2,Math.min(100,fuel))+"%";
  $("trafficBar").style.width=Math.max(2,Math.min(100,traffic))+"%";
  $("location").textContent=loc?`${fmt(loc.lat,4)}, ${fmt(loc.lon,4)}`:"—";
}
function renderMap(){
  const svg=$("map"), W=1000,H=600;
  const ns=network.nodes, lats=ns.map(n=>n.lat), lons=ns.map(n=>n.lon);
  mapBounds={minLat:Math.min(...lats),maxLat:Math.max(...lats),minLon:Math.min(...lons),maxLon:Math.max(...lons)};
  const xy=n=>({x:35+(n.lon-mapBounds.minLon)/(mapBounds.maxLon-mapBounds.minLon)*930,y:565-(n.lat-mapBounds.minLat)/(mapBounds.maxLat-mapBounds.minLat)*530});
  const lookup=new Map(ns.map(n=>[n.id,n]));
  $("roads").innerHTML=network.segments.map(s=>{const a=lookup.get(s.from),b=lookup.get(s.to);if(!a||!b)return"";const p=xy(a),q=xy(b);return `<line x1="${p.x}" y1="${p.y}" x2="${q.x}" y2="${q.y}"/>`}).join("");
  window.mapXY=xy; window.nodeLookup=lookup;
}
function drawRoute(r){
  $("routeLayer").innerHTML="";
  $("markers").innerHTML="";
  const path=r.nodes||[];
  for(let i=0;i<path.length-1;i++){
    const a=mapXY(nodeLookup.get(path[i])),b=mapXY(nodeLookup.get(path[i+1]));
    $("routeLayer").insertAdjacentHTML("beforeend",`<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`);
  }
  if(path.length){
    const a=mapXY(nodeLookup.get(path[0])),b=mapXY(nodeLookup.get(path[path.length-1]));
    $("markers").innerHTML=`<circle class="marker" cx="${a.x}" cy="${a.y}" r="7"/><circle class="destination-marker" cx="${b.x}" cy="${b.y}" r="8"/>`;
    animateCar(path);
  }
}
function animateCar(path){
  clearInterval(animationTimer);
  const car=$("car");car.classList.remove("hidden");
  let i=0,t=0;
  const move=()=>{if(!path.length)return;const n=nodeLookup.get(path[Math.min(i,path.length-1)]);const p=mapXY(n);car.setAttribute("cx",p.x);car.setAttribute("cy",p.y);i=(i+1)%path.length};
  move(); animationTimer=setInterval(move,900);
}
async function findRoute(hourOverride=null){
  if(!network)return;
  const vehicle=await get("/api/vehicle/"+encodeURIComponent(selectedVehicle));
  let source=vehicle.current_node;
  let destination=$("destination").value;
  let hour=hourOverride==null?Number($("hour").value):hourOverride;
  $("mapBadge").textContent="Analysing historical road behaviour…";
  try{
    const r=await get("/api/route",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({source,destination,priority:selectedPriority,hour})});
    routeData=r; rerouteHour=hour;
    drawRoute(r.recommended);
    renderRoute(r);
    $("headline").textContent="Your route is ready.";
    $("subline").textContent=`${selectedVehicle} · historical ${String(hour).padStart(2,"0")}:00 traffic · personalized multi-objective routing`;
    $("mapBadge").textContent=`Recommended path · ${r.recommended.distance_km} km · ${r.recommended.time_min} min`;
    $("rerouteBtn").classList.remove("hidden");
  }catch(e){$("mapBadge").textContent="Route unavailable — choose another destination";}
}
function renderRoute(r){
  $("routePanel").classList.remove("hidden");
  const x=r.recommended;
  $("routeTitle").textContent=`${x.distance_km} km smart route`;
  $("routeDistance").textContent=x.distance_km;
  $("routeTime").textContent=x.time_min;
  $("routeFuel").textContent=x.fuel_l;
  $("routeTraffic").textContent=x.traffic_pct+"%";
  $("explanation").innerHTML=(r.explanation||[]).map(s=>`<div>✓ ${s}</div>`).join("");
  $("whyText").innerHTML=(r.explanation||[]).map(s=>`<div>• ${s}</div>`).join("");
  $("alternatives").innerHTML=(r.alternatives||[]).map(a=>`<div class="alt"><span>${a.label}</span><b>${a.distance_km} km · ${a.time_min} min · ${a.fuel_l} L</b></div>`).join("");
}
function renderAnalytics(a){
  const max=Math.max(...a.hourly_traffic,0.01);
  $("trafficChart").innerHTML=a.hourly_traffic.map((v,i)=>`<div class="bar ${i===18?"hot":""}" style="height:${Math.max(3,v/max*95)}%" title="${i}:00 · ${(v*100).toFixed(0)}%"></div>`).join("");
  const items=Object.entries(a.fleet_mix).slice(0,5), total=items.reduce((s,x)=>s+x[1],0)||1;
  $("fleetChart").innerHTML=items.map(([k,v])=>`<div class="fleet-row"><label>${k}</label><i><b style="width:${v/Math.max(...items.map(x=>x[1]))*100}%"></b></i><span>${v}</span></div>`).join("");
}
$("routeBtn").onclick=()=>findRoute();
$("rerouteBtn").onclick=()=>{rerouteHour=Math.min(23,rerouteHour+2);$("hour").value=rerouteHour;$("hourLabel").textContent=String(rerouteHour).padStart(2,"0")+":00";findRoute(rerouteHour)};
$("destination").onchange=()=>{};
$("hour").oninput=e=>$("hourLabel").textContent=String(e.target.value).padStart(2,"0")+":00";
document.querySelectorAll(".mode").forEach(b=>b.onclick=()=>{document.querySelectorAll(".mode").forEach(x=>x.classList.remove("active"));b.classList.add("active");selectedPriority=b.dataset.priority});
boot();

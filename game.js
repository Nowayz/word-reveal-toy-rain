const wordEl = document.querySelector("#word");
const imageEl = document.querySelector("#revealImage");
const roundText = document.querySelector("#roundText");
const scoreText = document.querySelector("#scoreText");
const revealButton = document.querySelector("#revealButton");
const missButton = document.querySelector("#missButton");
const nextButton = document.querySelector("#nextButton");
const restartButton = document.querySelector("#restartButton");
const done = document.querySelector("#done");
const finalScore = document.querySelector("#finalScore");
const canvas = document.querySelector("#rain");
const ctx = canvas.getContext("2d");
const wordAudio = new Audio();

let deck = [];
let index = 0;
let score = 0;
let revealed = false;
let manifest = [];
let spriteImage = new Image();
let bodies = [];
let raf = 0;
let physics;
let walls = [];
let simulationStartedAt = 0;
let lastPhysicsTime = 0;
let physicsAccumulator = 0;
let releasingSprites = false;
let releaseComplete = null;

const FIXED_TIMESTEP = 1000 / 60;
const MAX_FRAME_DELTA = 100;

const MatterApi = () => window.Matter;

document.addEventListener("contextmenu", (event) => event.preventDefault());
document.addEventListener("dragstart", (event) => event.preventDefault());

function shuffle(items) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  rebuildWalls();
}

function current() {
  return deck[index];
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  let delta = b - a;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  return a + delta * t;
}

function updateHud() {
  roundText.textContent = `${Math.min(index + 1, deck.length)} / ${deck.length}`;
  scoreText.textContent = `Score ${score}`;
}

function showCard() {
  const item = current();
  revealed = false;
  clearPhysics();
  rebuildWalls();
  imageEl.classList.remove("show");
  imageEl.removeAttribute("src");
  imageEl.alt = "";
  wordEl.textContent = item.word;
  spriteImage = new Image();
  spriteImage.src = item.sprite;
  revealButton.disabled = false;
  missButton.disabled = false;
  nextButton.disabled = false;
  nextButton.hidden = true;
  updateHud();
}

function reveal(missed = false) {
  if (revealed) return;
  revealed = true;
  if (!missed) score += 1;
  const item = current();
  playWordAudio(item);
  imageEl.addEventListener("load", explodeSprites, { once: true });
  imageEl.src = item.image;
  imageEl.alt = item.word;
  requestAnimationFrame(() => imageEl.classList.add("show"));
  revealButton.disabled = true;
  missButton.disabled = true;
  nextButton.hidden = false;
  updateHud();
  if (imageEl.complete) requestAnimationFrame(explodeSprites);
}

function playWordAudio(item) {
  if (!item.audio) return;
  wordAudio.pause();
  wordAudio.currentTime = 0;
  wordAudio.src = item.audio;
  wordAudio.play().catch(() => {});
}

function setupPhysics() {
  if (!MatterApi()) return;
  const { Engine } = MatterApi();
  physics = Engine.create();
  physics.gravity.y = 1.15;
  rebuildWalls();
}

function rebuildWalls() {
  if (!physics || !MatterApi()) return;
  const { Bodies, Composite } = MatterApi();
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;

  if (walls.length) Composite.remove(physics.world, walls);
  walls = [
    Bodies.rectangle(width / 2, height + 32, width + 120, 64, { isStatic: true }),
    Bodies.rectangle(-32, height / 2, 64, height + 200, { isStatic: true }),
    Bodies.rectangle(width + 32, height / 2, 64, height + 200, { isStatic: true }),
    Bodies.rectangle(width / 2, -72, width + 120, 24, { isStatic: true }),
  ];
  Composite.add(physics.world, walls);
}

function clearPhysics() {
  cancelAnimationFrame(raf);
  raf = 0;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  releasingSprites = false;
  releaseComplete = null;
  if (!physics || !MatterApi()) {
    bodies = [];
    return;
  }
  const { Composite } = MatterApi();
  if (bodies.length) Composite.remove(physics.world, bodies.map((entry) => entry.body));
  bodies = [];
}

function explodeSprites() {
  if (!physics || !MatterApi()) return;
  clearPhysics();
  rebuildWalls();
  const { Bodies, Body, Composite, Vertices } = MatterApi();
  const item = current();
  const stageRect = canvas.getBoundingClientRect();
  const imageRect = imageEl.getBoundingClientRect();
  const centerX = imageRect.left - stageRect.left + imageRect.width / 2;
  const centerY = imageRect.top - stageRect.top + imageRect.height / 2;

  bodies = Array.from({ length: 82 }, (_, i) => {
    const size = 24 + Math.random() * 38;
    const angle = (Math.PI * 2 * i) / 82 + (Math.random() - 0.5) * 0.45;
    const radius = 16 + Math.random() * 42;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    const vertices = item.hitbox.map((point) => ({
      x: x + point.x * size,
      y: y + point.y * size,
    }));
    const body = Bodies.fromVertices(x, y, [Vertices.clockwiseSort(vertices)], {
      restitution: 0.94,
      friction: 0.12,
      frictionStatic: 0.18,
      frictionAir: 0.01,
      density: 0.0018,
    });
    Body.setInertia(body, body.inertia * 0.45);
    const speed = 12 + Math.random() * 13;
    Body.setVelocity(body, {
      x: Math.cos(angle) * speed,
      y: Math.sin(angle) * speed - 7 - Math.random() * 5,
    });
    Body.setAngularVelocity(body, (Math.random() - 0.5) * 0.95);
    return {
      body,
      size,
      torque: (Math.random() - 0.5) * 0.0007,
      previous: { x: body.position.x, y: body.position.y, angle: body.angle },
      current: { x: body.position.x, y: body.position.y, angle: body.angle },
    };
  });

  Composite.add(physics.world, bodies.map((entry) => entry.body));
  simulationStartedAt = performance.now();
  lastPhysicsTime = simulationStartedAt;
  physicsAccumulator = 0;
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(renderPhysics);
}

function releaseSprites(onComplete) {
  if (!physics || !MatterApi() || !bodies.length) {
    onComplete();
    return;
  }

  const { Body, Composite } = MatterApi();
  releasingSprites = true;
  releaseComplete = onComplete;
  nextButton.disabled = true;

  if (walls[0]) {
    Composite.remove(physics.world, walls[0]);
    walls = walls.slice(1);
  }

  for (const entry of bodies) {
    Body.setVelocity(entry.body, {
      x: entry.body.velocity.x * 0.45,
      y: Math.max(entry.body.velocity.y, 12 + Math.random() * 7),
    });
    Body.setAngularVelocity(entry.body, entry.body.angularVelocity + (Math.random() - 0.5) * 0.45);
  }

  if (!raf) {
    simulationStartedAt = performance.now();
    lastPhysicsTime = simulationStartedAt;
    physicsAccumulator = 0;
    raf = requestAnimationFrame(renderPhysics);
  }
}

function renderPhysics(now) {
  if (!physics || !MatterApi()) return;
  const { Engine, Composite } = MatterApi();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const elapsed = now - simulationStartedAt;
  const frameDelta = Math.min(MAX_FRAME_DELTA, now - lastPhysicsTime);
  lastPhysicsTime = now;
  physicsAccumulator += frameDelta;
  while (physicsAccumulator >= FIXED_TIMESTEP) {
    for (const entry of bodies) {
      entry.previous.x = entry.current.x;
      entry.previous.y = entry.current.y;
      entry.previous.angle = entry.current.angle;
    }
    Engine.update(physics, FIXED_TIMESTEP);
    for (const entry of bodies) {
      entry.current.x = entry.body.position.x;
      entry.current.y = entry.body.position.y;
      entry.current.angle = entry.body.angle;
    }
    physicsAccumulator -= FIXED_TIMESTEP;
  }
  ctx.clearRect(0, 0, w, h);
  const alpha = physicsAccumulator / FIXED_TIMESTEP;

  for (const entry of bodies) {
    if (elapsed < 1100) entry.body.torque += entry.torque;
    const x = lerp(entry.previous.x, entry.current.x, alpha);
    const y = lerp(entry.previous.y, entry.current.y, alpha);
    const angle = lerpAngle(entry.previous.angle, entry.current.angle, alpha);
    const size = entry.size;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(spriteImage, -size / 2, -size / 2, size, size);
    ctx.restore();
  }

  const expired = bodies.filter((entry) => entry.body.position.y > h + 220);
  if (expired.length) {
    Composite.remove(physics.world, expired.map((entry) => entry.body));
    bodies = bodies.filter((entry) => !expired.includes(entry));
  }

  if (bodies.length) raf = requestAnimationFrame(renderPhysics);
  else {
    raf = 0;
    ctx.clearRect(0, 0, w, h);
    if (releasingSprites && releaseComplete) {
      const complete = releaseComplete;
      releasingSprites = false;
      releaseComplete = null;
      complete();
    }
  }
}

function nextCard() {
  if (!revealed) return;
  if (releasingSprites) return;
  releaseSprites(advanceCard);
}

function advanceCard() {
  index += 1;
  if (index >= deck.length) {
    finalScore.textContent = `${score} out of ${deck.length}`;
    done.hidden = false;
    return;
  }
  showCard();
}

function restart() {
  deck = shuffle(manifest);
  index = 0;
  score = 0;
  done.hidden = true;
  showCard();
}

async function init() {
  resizeCanvas();
  setupPhysics();
  const response = await fetch("assets/manifest.json");
  manifest = await response.json();
  restart();
}

revealButton.addEventListener("click", () => reveal(false));
missButton.addEventListener("click", () => reveal(true));
nextButton.addEventListener("click", nextCard);
restartButton.addEventListener("click", restart);
window.addEventListener("resize", resizeCanvas);
window.addEventListener("keydown", (event) => {
  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    revealed ? nextCard() : reveal(false);
  }
});

init();

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
const stageEl = document.querySelector(".stage");
const canvas = document.querySelector("#rain");
const ctx = canvas.getContext("2d");

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
let floorWall = null;
let simulationStartedAt = 0;
let lastPhysicsTime = 0;
let physicsAccumulator = 0;
let releasingSprites = false;
let releaseComplete = null;
let cardVersion = 0;
let orientationEnabled = false;
let orientationPermissionStarted = false;
let orientationBeta = 90;
let orientationGamma = 0;
let orientationHeading = 0;
let orientationHasTilt = false;
let canvasResizeRaf = 0;
const letterActivationState = new WeakMap();
const preloadCache = new Map();
const audioBufferPromises = new Map();
const wordMeasureEl = document.createElement("div");
wordMeasureEl.className = "word word-measure";
wordMeasureEl.setAttribute("aria-hidden", "true");
document.body.appendChild(wordMeasureEl);

const FIXED_TIMESTEP = 1000 / 60;
const MAX_FRAME_DELTA = 100;
const MAX_WORD_FONT_SIZE = 136;
const MIN_WORD_FONT_SIZE = 24;
const ACTIVE_LETTER_SCALE = 1.25;
const BASE_GRAVITY_Y = 0.8;
const ORIENTED_GRAVITY_SCALE = 0.8;
const AUDIO_VISUAL_SYNC_FALLBACK_MS = 650;
const EMPTY_IMAGE =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==";

const MatterApi = () => window.Matter;
const AudioContextApi = () => window.AudioContext || window.webkitAudioContext;
let audioContext = null;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function getAudioContext() {
  if (audioContext || !AudioContextApi()) return audioContext;
  try {
    audioContext = new (AudioContextApi())();
  } catch {
    audioContext = null;
  }
  return audioContext;
}

async function resumeAudioContext() {
  const context = getAudioContext();
  if (!context) return null;
  if (context.state === "suspended") {
    try {
      await context.resume();
    } catch {
      return null;
    }
  }
  return context;
}

function decodeAudioData(context, arrayBuffer) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (buffer) => {
      if (settled) return;
      settled = true;
      resolve(buffer || null);
    };

    const maybePromise = context.decodeAudioData(arrayBuffer, finish, () => finish(null));
    if (maybePromise && typeof maybePromise.then === "function") {
      maybePromise.then(finish, () => finish(null));
    }
  });
}

function preloadAudio(src) {
  const context = getAudioContext();
  if (!src || !context) return Promise.resolve(null);
  if (!audioBufferPromises.has(src)) {
    audioBufferPromises.set(
      src,
      fetch(src)
        .then((response) => {
          if (!response.ok) throw new Error(`Unable to load audio: ${src}`);
          return response.arrayBuffer();
        })
        .then((arrayBuffer) => decodeAudioData(context, arrayBuffer))
        .catch(() => null)
    );
  }
  return audioBufferPromises.get(src);
}

async function playAudio(src) {
  const context = await resumeAudioContext();
  const buffer = await preloadAudio(src);
  if (!context || !buffer) return false;

  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  try {
    source.start(0);
    return true;
  } catch {
    return false;
  }
}

function unlockAudio() {
  resumeAudioContext();
}

function unlockInputFeatures() {
  unlockAudio();
  setupOrientation();
}

function waitForImageLoad(image) {
  if (image.complete && image.naturalWidth) return Promise.resolve();
  return new Promise((resolve) => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", resolve, { once: true });
  });
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function letterAudioSrc(letter) {
  const normalized = letter.trim().toLowerCase();
  return /^[a-z]$/.test(normalized) ? `assets/audio/${normalized}.opus` : null;
}

function preloadLetterAudio(word) {
  for (const letter of Array.from(word)) {
    preloadAudio(letterAudioSrc(letter));
  }
}

function playLetterAudio(letter) {
  playAudio(letterAudioSrc(letter));
}

function activateLetter(letter, letterCell) {
  const state = letterActivationState.get(letterCell) || { locked: false };
  if (state.locked) return;
  state.locked = true;
  letterActivationState.set(letterCell, state);

  letterCell.classList.add("active");
  window.clearTimeout(state.timeout);
  state.timeout = window.setTimeout(() => {
    letterCell.classList.remove("active");
    state.locked = false;
  }, 1000);
  letterActivationState.set(letterCell, state);

  playLetterAudio(letter);
}

function renderWord(word) {
  renderWordInto(wordEl, word);
  preloadLetterAudio(word);
  const cached = preloadCache.get(word);
  if (cached && cached.fontSize) {
    wordEl.style.setProperty("--word-font-size", `${cached.fontSize}px`);
  }
  requestAnimationFrame(() => {
    fitAndSyncWord();
    requestAnimationFrame(fitAndSyncWord);
  });
  if (document.fonts) {
    document.fonts.ready.then(fitAndSyncWord);
  }
}

function renderWordInto(container, word) {
  container.replaceChildren();
  const fragment = document.createDocumentFragment();
  const tokens = word.match(/\S+|\s+/g) || [];

  for (const token of tokens) {
    if (/^\s+$/.test(token)) {
      const space = document.createElement("span");
      space.className = "letter-space";
      space.innerHTML = "&nbsp;";
      fragment.appendChild(space);
      continue;
    }

    const wordToken = document.createElement("span");
    wordToken.className = "word-token";

    for (const letter of Array.from(token)) {
      const letterCell = document.createElement("span");
      const glyph = document.createElement("span");
      const target = document.createElement("span");
      letterCell.className = "letter-cell";
      glyph.className = "letter-glyph";
      target.className = "letter-target";
      target.setAttribute("role", "button");
      target.setAttribute("tabindex", "0");
      glyph.textContent = letter;

      if (container === wordEl) {
        target.addEventListener("pointerdown", (event) => {
          event.preventDefault();
          // Capture pointer so dragging off the letter still activates it
          target.setPointerCapture(event.pointerId);
          activateLetter(letter, letterCell);
        });
        target.addEventListener("click", (event) => {
          event.preventDefault();
        });
        target.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activateLetter(letter, letterCell);
          }
        });
      }
      letterCell.appendChild(glyph);
      letterCell.appendChild(target);
      wordToken.appendChild(letterCell);
    }

    fragment.appendChild(wordToken);
  }

  container.appendChild(fragment);
}

function fitWord(container = wordEl) {
  if (!stageEl || !container.clientWidth || !container.children.length) return MIN_WORD_FONT_SIZE;

  const stageRect = stageEl.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();
  const wordRect = container.getBoundingClientRect();
  const styles = window.getComputedStyle(container);
  const horizontalPadding =
    parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
  const verticalPadding =
    parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
  const availableWidth = Math.max(1, container.clientWidth - horizontalPadding);
  const availableHeight = Math.max(
    72,
    stageRect.bottom - wordRect.top - verticalPadding - 32
  );

  const fits = () => {
    const activeScaleBuffer = ACTIVE_LETTER_SCALE + 0.04;
    if (container.scrollWidth > container.clientWidth + 1) return false;
    if (container.scrollHeight > availableHeight + verticalPadding + 1) return false;

    for (const token of container.querySelectorAll(".word-token")) {
      const rect = token.getBoundingClientRect();
      if (rect.width * activeScaleBuffer > availableWidth) return false;
      if (
        rect.left < containerRect.left + parseFloat(styles.paddingLeft) - 1 ||
        rect.right > containerRect.right - parseFloat(styles.paddingRight) + 1
      ) {
        return false;
      }
    }

    return true;
  };

  let low = MIN_WORD_FONT_SIZE;
  let high = MAX_WORD_FONT_SIZE;
  let best = low;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    container.style.setProperty("--word-font-size", `${mid}px`);

    if (fits()) {
      best = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  container.style.setProperty("--word-font-size", `${best}px`);
  return best;
}

function fitAndSyncWord() {
  const fontSize = fitWord();
  scheduleCanvasResize();
  return fontSize;
}

function syncLayoutToViewport() {
  fitAndSyncWord();
  scheduleCanvasResize();
}

function preloadAsset(src, kind = "image") {
  if (!src) return;
  if (kind === "audio") {
    preloadAudio(src);
    return;
  }
  const image = new Image();
  image.decoding = "async";
  image.src = src;
  if (image.decode) image.decode().catch(() => {});
}

function preloadCard(card) {
  if (!card || preloadCache.has(card.word)) return;
  const entry = { fontSize: null };
  preloadCache.set(card.word, entry);

  preloadAsset(card.image);
  preloadAsset(card.sprite);
  preloadAsset(card.audio, "audio");
  preloadLetterAudio(card.word);

  requestAnimationFrame(() => {
    wordMeasureEl.style.width = `${wordEl.clientWidth || stageEl.clientWidth}px`;
    renderWordInto(wordMeasureEl, card.word);
    entry.fontSize = fitWord(wordMeasureEl);
  });
}

function preloadNextCard() {
  preloadCard(deck[index + 1]);
}

function blockContextMenu(event) {
  event.preventDefault();
  event.stopPropagation();
  if (event.stopImmediatePropagation) {
    event.stopImmediatePropagation();
  }
  return false;
}

function blockNonPrimaryPointer(event) {
  const isRightClick = event.button === 2;
  const isTouchOrPen = event.pointerType === "touch" || event.pointerType === "pen";
  const isButton =
    event.target instanceof Element &&
    (event.target.closest("button") !== null ||
      event.target.closest(".letter-target") !== null);

  if (isRightClick || (isTouchOrPen && !isButton)) {
    return blockContextMenu(event);
  }
  return false;
}

window.addEventListener("contextmenu", blockContextMenu, { capture: true });
window.addEventListener("auxclick", blockContextMenu, { capture: true });
window.addEventListener("pointerdown", blockNonPrimaryPointer, { capture: true, passive: false });
window.addEventListener("pointerdown", unlockInputFeatures, { capture: true });
window.addEventListener("keydown", unlockInputFeatures, { capture: true });
window.addEventListener("selectstart", blockContextMenu, { capture: true });
window.addEventListener("dragstart", (event) => event.preventDefault());

document.addEventListener("contextmenu", blockContextMenu, { capture: true });
document.addEventListener("auxclick", blockContextMenu, { capture: true });

window.oncontextmenu = blockContextMenu;
document.body && (document.body.oncontextmenu = blockContextMenu);
document.documentElement && (document.documentElement.oncontextmenu = blockContextMenu);

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
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  rebuildWalls();
}

function scheduleCanvasResize() {
  if (canvasResizeRaf) return;
  canvasResizeRaf = requestAnimationFrame(() => {
    canvasResizeRaf = 0;
    resizeCanvas();
  });
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
  cardVersion += 1;
  const item = current();
  revealed = false;
  preloadCard(item);
  clearPhysics();
  rebuildWalls();
  imageEl.classList.remove("show");
  imageEl.src = EMPTY_IMAGE;
  imageEl.alt = "";
  renderWord(item.word);
  spriteImage = new Image();
  spriteImage.src = item.sprite;
  revealButton.disabled = false;
  missButton.disabled = false;
  nextButton.disabled = true;
  nextButton.hidden = true;
  updateHud();
  preloadNextCard();
}

function reveal(missed = false) {
  if (revealed) return;
  revealed = true;
  if (!missed) score += 1;
  const item = current();
  const revealVersion = cardVersion;
  imageEl.src = item.image;
  imageEl.alt = item.word;
  revealButton.disabled = true;
  missButton.disabled = true;
  nextButton.hidden = false;
  nextButton.disabled = true;
  updateHud();

  Promise.all([
    waitForImageLoad(imageEl),
    Promise.race([playWordAudio(item), delay(AUDIO_VISUAL_SYNC_FALLBACK_MS)]),
  ]).then(() => {
    if (revealVersion !== cardVersion || !revealed) return;
    requestAnimationFrame(() => {
      imageEl.classList.add("show");
      explodeSprites(revealVersion);
    });
  });
}

function playWordAudio(item) {
  if (!item.audio) return Promise.resolve(false);
  return playAudio(item.audio);
}

function setupPhysics() {
  if (!MatterApi()) return;
  const { Engine } = MatterApi();
  physics = Engine.create();
  physics.gravity.y = BASE_GRAVITY_Y;
  rebuildWalls();
}

function handleOrientation(event) {
  if (!Number.isFinite(event.beta) || !Number.isFinite(event.gamma)) return;

  const heading =
    Number.isFinite(event.webkitCompassHeading) ? event.webkitCompassHeading : event.alpha;
  orientationBeta = event.beta;
  orientationGamma = event.gamma;
  if (Number.isFinite(heading)) orientationHeading = heading;
  orientationHasTilt = true;
}

function getOrientedGravity() {
  if (!orientationHasTilt) {
    return { x: 0, y: BASE_GRAVITY_Y };
  }

  const beta = (clamp(orientationBeta, -90, 90) * Math.PI) / 180;
  const gamma = (clamp(orientationGamma, -90, 90) * Math.PI) / 180;
  const x = clamp(Math.sin(gamma), -1, 1) * ORIENTED_GRAVITY_SCALE;
  const y = clamp(Math.sin(beta), -1, 1) * ORIENTED_GRAVITY_SCALE;

  return { x, y };
}

function setupOrientation() {
  if (
    orientationEnabled ||
    orientationPermissionStarted ||
    !("DeviceOrientationEvent" in window)
  ) {
    return;
  }
  orientationPermissionStarted = true;

  const enableOrientation = () => {
    if (orientationEnabled) return;
    window.addEventListener("deviceorientationabsolute", handleOrientation);
    window.addEventListener("deviceorientation", handleOrientation);
    orientationEnabled = true;
  };

  if (typeof DeviceOrientationEvent.requestPermission === "function") {
    DeviceOrientationEvent.requestPermission()
      .then((response) => {
        if (response === "granted") enableOrientation();
      })
      .catch(() => {
        orientationPermissionStarted = false;
      });
  } else {
    enableOrientation();
  }
}

function shrinkBoundingBox(points, insetPx) {
  const inset = Math.max(0, insetPx);
  if (!points.length || !inset) {
    return points;
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
  }
  const width = maxX - minX;
  const height = maxY - minY;
  if (width <= inset * 2 || height <= inset * 2) {
    return points;
  }
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const scaleX = (width - inset * 2) / width;
  const scaleY = (height - inset * 2) / height;
  return points.map((point) => ({
    x: centerX + (point.x - centerX) * scaleX,
    y: centerY + (point.y - centerY) * scaleY,
  }));
}

function rebuildWalls() {
  if (!physics || !MatterApi()) return;
  const { Bodies, Composite } = MatterApi();
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;

  if (walls.length) Composite.remove(physics.world, walls);
  floorWall = Bodies.rectangle(width / 2, height + 32, width + 120, 64, { isStatic: true });
  walls = [
    floorWall,
    Bodies.rectangle(-32, height / 2, 64, height + 200, { isStatic: true }),
    Bodies.rectangle(width + 32, height / 2, 64, height + 200, { isStatic: true }),
    Bodies.rectangle(width / 2, -72, width + 120, 24, { isStatic: true }),
  ];
  Composite.add(physics.world, walls);
}

function resetWalls() {
  if (!physics || !MatterApi()) return;
  const { Composite } = MatterApi();
  if (walls.length) {
    Composite.remove(physics.world, walls);
  }
  walls = [];
  floorWall = null;
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

function explodeSprites(expectedVersion) {
  if (expectedVersion !== undefined && expectedVersion !== cardVersion) return;
  if (!physics || !MatterApi()) {
    nextButton.disabled = false;
    return;
  }
  clearPhysics();
  if (expectedVersion !== undefined && expectedVersion !== cardVersion) return;
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
    const scaledPoints = item.hitbox.map((point) => ({
      x: point.x * size,
      y: point.y * size,
    }));
    const insetPoints = shrinkBoundingBox(scaledPoints, 1);
    const vertices = insetPoints.map((point) => ({
      x: x + point.x,
      y: y + point.y,
    }));
    const body = Bodies.fromVertices(x, y, [Vertices.clockwiseSort(vertices)], {
      restitution: 0.9,
      friction: 0.5,
      frictionStatic: 0.85,
      frictionAir: 0.0,
      density: 0.0018,
    });
    Body.setInertia(body, body.inertia * 2.5);
    const speed = 8 + Math.random() * 8;
    Body.setVelocity(body, {
      x: Math.cos(angle) * speed,
      y: Math.sin(angle) * speed - 7 - Math.random() * 5,
    });
    Body.setAngularVelocity(body, (Math.random() - 0.5) * 0.15);
    return {
      body,
      size,
      torque: (Math.random() - 0.5) * 0.00006,
      previous: { x: body.position.x, y: body.position.y, angle: body.angle },
      current: { x: body.position.x, y: body.position.y, angle: body.angle },
    };
  });

  Composite.add(physics.world, bodies.map((entry) => entry.body));
  nextButton.disabled = false;
  simulationStartedAt = performance.now();
  lastPhysicsTime = simulationStartedAt;
  physicsAccumulator = 0;
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(renderPhysics);
}

function releaseSprites(onComplete) {
  if (!physics || !MatterApi() || !bodies.length) {
    rebuildWalls();
    onComplete();
    return;
  }

  const { Body, Composite } = MatterApi();
  releasingSprites = true;
  releaseComplete = onComplete;
  nextButton.disabled = true;

  resetWalls();

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
    const gravity = getOrientedGravity();
    physics.gravity.x = gravity.x;
    physics.gravity.y = gravity.y;
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
    if (elapsed < 600) entry.body.torque += entry.torque;
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
      rebuildWalls();
      complete();
    }
  }
}

function nextCard() {
  if (!revealed) return;
  if (releasingSprites) return;
  if (nextButton.hidden || nextButton.disabled) return;
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

function bindTouchButton(button, handler) {
  let suppressClick = false;
  button.addEventListener("pointerup", (event) => {
    if (event.button !== 0 && event.button !== undefined) return;
    event.preventDefault();
    suppressClick = true;
    handler();
    window.setTimeout(() => {
      suppressClick = false;
    }, 0);
  });
  button.addEventListener("click", (event) => {
    if (suppressClick) {
      suppressClick = false;
      return;
    }
    handler();
  });
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
bindTouchButton(revealButton, () => reveal(false));
bindTouchButton(missButton, () => reveal(true));
bindTouchButton(nextButton, () => nextCard());
restartButton.addEventListener("click", restart);
window.addEventListener("resize", syncLayoutToViewport);
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", syncLayoutToViewport);
  window.visualViewport.addEventListener("scroll", scheduleCanvasResize);
}
window.addEventListener("keydown", (event) => {
  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    revealed ? nextCard() : reveal(false);
  }
});

if (window.ResizeObserver && stageEl) {
  const stageResizeObserver = new ResizeObserver(scheduleCanvasResize);
  stageResizeObserver.observe(stageEl);
}

init();

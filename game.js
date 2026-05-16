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
let floorWall = null;
let simulationStartedAt = 0;
let lastPhysicsTime = 0;
let physicsAccumulator = 0;
let releasingSprites = false;
let releaseComplete = null;
let cardVersion = 0;
const letterActivationState = new WeakMap();
const preloadCache = new Map();
const letterAudioCache = new Map();
const wordMeasureEl = document.createElement("div");
wordMeasureEl.className = "word word-measure";
wordMeasureEl.setAttribute("aria-hidden", "true");
document.body.appendChild(wordMeasureEl);

const FIXED_TIMESTEP = 1000 / 60;
const MAX_FRAME_DELTA = 100;
const MAX_WORD_FONT_SIZE = 136;
const MIN_WORD_FONT_SIZE = 24;
const ACTIVE_LETTER_SCALE = 1.25;

const MatterApi = () => window.Matter;

function letterNameForSpoken(letter) {
  const normalized = letter.trim().toLowerCase();
  if (!normalized) return "";
  if (normalized.length === 1 && normalized >= "a" && normalized <= "z") {
    return `Letter ${normalized.toUpperCase()}`;
  }
  if (/[0-9]/.test(normalized)) {
    return `Number ${normalized}`;
  }
  return normalized;
}

function speakLetter(letter) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(letterNameForSpoken(letter));
  utterance.rate = 1;
  window.speechSynthesis.speak(utterance);
}

function getLetterAudio(letter) {
  const normalized = letter.trim().toLowerCase();
  if (!/^[a-z]$/.test(normalized)) return null;
  if (!letterAudioCache.has(normalized)) {
    const audio = new Audio(`assets/audio/${normalized}.opus`);
    audio.preload = "auto";
    letterAudioCache.set(normalized, audio);
  }
  return letterAudioCache.get(normalized);
}

function preloadLetterAudio(word) {
  for (const letter of Array.from(word)) {
    getLetterAudio(letter);
  }
}

function playLetterAudio(letter) {
  const audio = getLetterAudio(letter);
  if (!audio) {
    speakLetter(letter);
    return;
  }

  audio.pause();
  audio.currentTime = 0;
  audio.play().catch(() => speakLetter(letter));
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
    fitWord();
    requestAnimationFrame(fitWord);
  });
  if (document.fonts) {
    document.fonts.ready.then(fitWord);
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
        const onActivate = (event) => {
          event.preventDefault();
          activateLetter(letter, letterCell);
        };

        target.addEventListener("pointerup", onActivate);
        target.addEventListener("click", onActivate);
        target.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            onActivate(event);
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
  const stage = document.querySelector(".stage");
  if (!stage || !container.clientWidth || !container.children.length) return MIN_WORD_FONT_SIZE;

  const stageRect = stage.getBoundingClientRect();
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
      if (rect.left < stageRect.left + 16 || rect.right > stageRect.right - 16) {
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

function preloadAsset(src, kind = "image") {
  if (!src) return;
  if (kind === "audio") {
    const audio = new Audio();
    audio.preload = "auto";
    audio.src = src;
    audio.load();
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
window.addEventListener("mousedown", blockNonPrimaryPointer, { capture: true });
window.addEventListener("pointerdown", blockNonPrimaryPointer, { capture: true, passive: false });
window.addEventListener("pointerup", blockNonPrimaryPointer, { capture: true, passive: false });
window.addEventListener("pointercancel", blockContextMenu, { capture: true, passive: false });
window.addEventListener("selectstart", blockContextMenu, { capture: true });
window.addEventListener("dragstart", (event) => event.preventDefault());

document.addEventListener("contextmenu", blockContextMenu, { capture: true });
document.addEventListener("auxclick", blockContextMenu, { capture: true });
document.addEventListener("mousedown", blockNonPrimaryPointer, { capture: true });
document.addEventListener("pointerdown", blockNonPrimaryPointer, { capture: true, passive: false });
document.addEventListener("pointerup", blockNonPrimaryPointer, { capture: true, passive: false });
document.addEventListener("pointercancel", blockContextMenu, { capture: true, passive: false });
document.addEventListener("selectstart", blockContextMenu, { capture: true });

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
  cardVersion += 1;
  const item = current();
  revealed = false;
  clearPhysics();
  rebuildWalls();
  imageEl.classList.remove("show");
  imageEl.removeAttribute("src");
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
  let spritesStarted = false;
  const startSprites = () => {
    if (spritesStarted || revealVersion !== cardVersion || !revealed) return;
    spritesStarted = true;
    explodeSprites(revealVersion);
  };
  playWordAudio(item);
  imageEl.addEventListener("load", startSprites, { once: true });
  imageEl.src = item.image;
  imageEl.alt = item.word;
  requestAnimationFrame(() => imageEl.classList.add("show"));
  revealButton.disabled = true;
  missButton.disabled = true;
  nextButton.hidden = false;
  nextButton.disabled = true;
  updateHud();
  if (imageEl.complete && imageEl.naturalWidth) requestAnimationFrame(startSprites);
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
      restitution: 0.98,
      friction: 0.32,
      frictionStatic: 0.72,
      frictionAir: 0.014,
      density: 0.0018,
    });
    Body.setInertia(body, body.inertia * 1.8);
    const speed = 12 + Math.random() * 13;
    Body.setVelocity(body, {
      x: Math.cos(angle) * speed,
      y: Math.sin(angle) * speed - 7 - Math.random() * 5,
    });
    Body.setAngularVelocity(body, (Math.random() - 0.5) * 0.35);
    return {
      body,
      size,
      torque: (Math.random() - 0.5) * 0.00018,
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
window.addEventListener("resize", resizeCanvas);
window.addEventListener("resize", fitWord);
window.addEventListener("keydown", (event) => {
  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    revealed ? nextCard() : reveal(false);
  }
});

init();

// Script to append frontend scenarios to scenarios.js cleanly
const fs = require('fs');
const path = require('path');

const targetFile = path.join(__dirname, '..', 'src', 'data', 'scenarios.js');

const frontendScenarios = `
  // ─── FRONTEND SCENARIOS ───────────────────────────────────────────────────
  {
    id: 'css-flexbox-bug',
    type: 'frontend',
    title: 'Broken Flex Layout',
    description: 'Three nav items should sit side-by-side in a horizontal row, but they are stacking vertically. Fix the CSS so the layout displays correctly.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar {
      display: flex;
      /* BUG: items are stacking because of wrong flex-direction */
      flex-direction: column;
      gap: 12px;
      background: #1e1e1e;
      padding: 16px 32px;
      border-radius: 12px;
      border: 1px solid #333;
    }
    .nav-item {
      padding: 10px 24px;
      background: #7c3aed;
      border-radius: 8px;
      font-weight: bold;
      color: #fff;
      text-align: center;
    }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>\`
      }
    },
    constraints: ['Items must use flexbox.', "Change flex-direction to 'row'."],
    desiredOutput: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar { display: flex; flex-direction: row; gap: 12px; background: #1e1e1e; padding: 16px 32px; border-radius: 12px; border: 1px solid #333; }
    .nav-item { padding: 10px 24px; background: #7c3aed; border-radius: 8px; font-weight: bold; color: #fff; text-align: center; }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>\`,
    previewCheck: \`
      try {
        const navbar = document.querySelector('.navbar');
        const style = window.getComputedStyle(navbar);
        const dir = style.flexDirection;
        return dir === 'row' || dir === 'row-reverse';
      } catch(e) { return false; }
    \`
  },
  {
    id: 'js-counter-bug',
    type: 'frontend',
    title: 'Click Counter Bug',
    description: 'A counter button should increment the displayed number each time it is clicked. The counter is stuck at 0. Find and fix the two bugs.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
    button:hover { background: #6d28d9; }
  </style>
</head>
<body>
  <div id="count">0</div>
  <!-- BUG 1: button type submit causes page reload -->
  <button type="submit" id="btn">Click Me</button>
  <script>
    let count = 0;
    const display = document.getElementById('count');
    // BUG 2: wrong element ID - 'button' does not exist, should be 'btn'
    document.getElementById('button').addEventListener('click', () => {
      count++;
      display.textContent = count;
    });
  <\\/script>
</body>
</html>\`
      }
    },
    constraints: ["Fix button type to 'button'.", "Fix getElementById to use 'btn'."],
    desiredOutput: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
  </style>
</head>
<body>
  <div id="count">1</div>
  <button type="button" id="btn">Click Me</button>
</body>
</html>\`,
    previewCheck: \`
      try {
        const btn = document.getElementById('btn');
        if (!btn) return false;
        const correctType = btn.type === 'button';
        btn.click();
        const count = parseInt(document.getElementById('count').textContent, 10);
        return correctType && count > 0;
      } catch(e) { return false; }
    \`
  },
  {
    id: 'dom-color-bug',
    type: 'frontend',
    title: 'Dynamic Color Picker',
    description: 'Clicking a color swatch should change the background of the box below it. Nothing happens. Fix the JavaScript typo.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; transition: transform 0.2s; }
    .swatch:hover { transform: scale(1.1); border-color: #fff; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #1e1e1e; border: 2px solid #333; transition: background 0.3s; }
    p { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" data-color="#7c3aed" style="background:#7c3aed"></div>
    <div class="swatch" data-color="#ec4899" style="background:#ec4899"></div>
    <div class="swatch" data-color="#10b981" style="background:#10b981"></div>
    <div class="swatch" data-color="#f59e0b" style="background:#f59e0b"></div>
  </div>
  <div id="color-box"></div>
  <p>Click a color swatch above</p>
  <script>
    const swatches = document.querySelectorAll('.swatch');
    const box = document.getElementById('color-box');
    swatches.forEach(swatch => {
      swatch.addEventListener('click', () => {
        // BUG: dataset.colour does not exist (typo - should be dataset.color)
        box.style.backgroundColor = swatch.dataset.colour;
      });
    });
  <\\/script>
</body>
</html>\`
      }
    },
    constraints: ["Fix 'dataset.colour' typo to 'dataset.color'.", "Set box.style.backgroundColor."],
    desiredOutput: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #7c3aed; border: 2px solid #7c3aed; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" style="background:#7c3aed"></div>
    <div class="swatch" style="background:#ec4899"></div>
  </div>
  <div id="color-box" style="background:#7c3aed"></div>
</body>
</html>\`,
    previewCheck: \`
      try {
        const swatch = document.querySelector('.swatch');
        const box = document.getElementById('color-box');
        if (!swatch || !box) return false;
        swatch.click();
        const bg = box.style.backgroundColor || box.style.background;
        return bg !== '' && bg !== undefined && bg !== 'rgba(0, 0, 0, 0)';
      } catch(e) { return false; }
    \`
  },
  {
    id: 'css-card-hover-bug',
    type: 'frontend',
    title: 'Card Hover Effect Bug',
    description: 'Three product cards should lift and glow on hover. The hover animation is completely broken. Fix the two CSS bugs.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card {
      width: 180px; padding: 28px 20px; background: #1a1a2e;
      border: 1px solid #333; border-radius: 16px; text-align: center; cursor: pointer;
      /* BUG 1: transition is missing */
    }
    /* BUG 2: wrong selector - '.cards:hover' should be '.card:hover' */
    .cards:hover {
      transform: translateY(-12px);
      box-shadow: 0 20px 60px rgba(124, 58, 237, 0.5);
      border-color: #7c3aed;
    }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>\`
      }
    },
    constraints: ["Add 'transition: transform 0.3s, box-shadow 0.3s' to .card.", "Fix hover selector from '.cards:hover' to '.card:hover'."],
    desiredOutput: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card { width: 180px; padding: 28px 20px; background: #1a1a2e; border: 1px solid #7c3aed; border-radius: 16px; text-align: center; cursor: pointer; transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s; transform: translateY(-12px); box-shadow: 0 20px 60px rgba(124,58,237,0.5); }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>\`,
    previewCheck: \`
      try {
        const card = document.querySelector('.card');
        if (!card) return false;
        const style = window.getComputedStyle(card);
        const hasTransition = style.transitionDuration && style.transitionDuration !== '0s';
        let hasHoverRule = false;
        for (const sheet of document.styleSheets) {
          try {
            for (const rule of sheet.cssRules) {
              if (rule.selectorText && rule.selectorText.includes('.card:hover')) hasHoverRule = true;
            }
          } catch(e) {}
        }
        return hasTransition && hasHoverRule;
      } catch(e) { return false; }
    \`
  },
  {
    id: 'js-todo-list-bug',
    type: 'frontend',
    title: 'Todo List Builder Bug',
    description: 'A todo app should add new tasks when clicking "Add". Items are not appearing and the input is never cleared. Fix the three JavaScript bugs.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; outline: none; }
    input:focus { border-color: #7c3aed; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; font-size: 14px; }
    button:hover { background: #6d28d9; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list"></ul>
  </div>
  <script>
    const input = document.getElementById('todo-input');
    const list = document.getElementById('todo-list');
    const btn = document.getElementById('add-btn');
    btn.addEventListener('click', addTodo);
    function addTodo() {
      const text = input.value;
      // BUG 1: No empty check - should guard with: if (!text.trim()) return;
      const li = document.createElement('li');
      // BUG 2: Using innerHTML with raw text - should use textContent
      li.innerHTML = text;
      list.appendChild(li);
      // BUG 3: Input is never cleared - should add: input.value = '';
    }
  <\\/script>
</body>
</html>\`
      }
    },
    constraints: ['Guard against empty input with trim() check.', 'Use textContent not innerHTML.', 'Clear input.value after adding.'],
    desiredOutput: \`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list">
      <li>Buy groceries</li>
      <li>Write clean code</li>
    </ul>
  </div>
</body>
</html>\`,
    previewCheck: \`
      try {
        const input = document.getElementById('todo-input');
        const btn = document.getElementById('add-btn');
        const list = document.getElementById('todo-list');
        if (!input || !btn || !list) return false;
        input.value = 'Test Task';
        btn.click();
        const items = list.querySelectorAll('li');
        const added = items.length > 0;
        const cleared = input.value === '';
        const safeContent = items.length > 0 && items[items.length-1].textContent.trim() === 'Test Task';
        return added && cleared && safeContent;
      } catch(e) { return false; }
    \`
  }
`;

// Read original file
let content = fs.readFileSync(targetFile, 'utf8');

// Remove trailing whitespace and the closing ];
content = content.trimEnd();
if (content.endsWith('];')) {
    content = content.slice(0, -2).trimEnd();
}

// Append the new scenarios and close the array
const newContent = content + ',\n' + frontendScenarios + '\n];\n';

fs.writeFileSync(targetFile, newContent, 'utf8');
console.log('✅ Frontend scenarios appended successfully!');
console.log('File size:', fs.statSync(targetFile).size, 'bytes');

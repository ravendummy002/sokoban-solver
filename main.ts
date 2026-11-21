// Sokoban solver for make.fun's sokoban-solver-01-run-20261121.
//
// A level is a hallway with rooms off it, one door each, and a box pushed out
// of a room into the hallway can never come back — so every room is a problem
// of its own, and the level's push-optimal count P* is the sum of the rooms'
// optima. This solver splits the level into rooms (the hallway's squares are
// dead: no push from one reaches a goal, so cutting them apart separates the
// rooms), solves each room push-optimally with A* over pushes — a state is the
// boxes plus the player's reachable region, the estimate is the cheapest
// matching of boxes to goals over walking distances, and dead squares plus
// frozen 2 x 2 blocks prune positions no push can save — then walks the player
// from room to room between solutions. Walking costs no pushes, so the rooms'
// optima add up to P*.
//
// Deterministic by construction: the budgets count work (states created), never
// the clock; there is no randomness, and every tie is broken by a fixed order,
// so two runs write identical bytes.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states the optimal search may create for one room, and the greedy fallback's.
 *  250,000 matches the generator's own SEARCH_STATES: every room it keeps is solvable within that many states,
 *  and in practice a room takes a few thousand — so the cap is a safety net, never the plan. */
const ASTAR_BUDGET = 250_000;
const GREEDY_BUDGET = 250_000;

/** [dx, dy, letter]: up, down, left, right — this fixed order decides every tie. */
const DIRS: ReadonlyArray<readonly [number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];
/** The opposite direction of each DIRS entry. */
const OPP = [1, 0, 3, 2];

/** Powers of 1024, for packing up to four box squares (and a region's lowest) into one exact number. */
const POW = [1, 1024, 1048576, 1073741824, 1099511627776];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  boxes: number[];
  goals: number[];
}

interface RoomSolution {
  pushes: number;
  moves: string;
}

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/** Each square's four neighbours, -1 past the board's edge, so no walk ever steps off the array. */
function neighborTable(w: number, h: number): Int32Array {
  const nbr = new Int32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = y * w + x;
      nbr[p * 4 + 0] = y > 0 ? p - w : -1;
      nbr[p * 4 + 1] = y < h - 1 ? p + w : -1;
      nbr[p * 4 + 2] = x > 0 ? p - 1 : -1;
      nbr[p * 4 + 3] = x < w - 1 ? p + 1 : -1;
    }
  }
  return nbr;
}

/**
 * The squares a box can still be pushed from onto some goal, found by pulling
 * back from the goals: a square y is live when some live square x = y + d can
 * be reached by pushing the box from y, which needs the player to stand at
 * y - d, so both y and y - d must be floor. A square outside this set can
 * never hold a box on its way to a goal, whatever the other boxes do.
 */
function liveSquares(wall: Uint8Array, nbr: Int32Array, goals: number[]): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue = goals.slice();
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi] as number;
    for (let k = 0; k < 4; k++) {
      const y = nbr[x * 4 + OPP[k]] as number; // the square the box was pushed from
      if (y < 0 || wall[y] || live[y]) continue;
      const behind = nbr[y * 4 + OPP[k]] as number; // where the player must stand
      if (behind < 0 || wall[behind]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/** Each goal's walking distance to every square, over open floor (walls only, no boxes). */
function goalDistances(wall: Uint8Array, nbr: Int32Array, goals: number[]): Int32Array[] {
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi] as number;
      for (let k = 0; k < 4; k++) {
        const q = nbr[p * 4 + k] as number;
        if (q < 0 || wall[q] || dist[q] >= 0) continue;
        dist[q] = dist[p] + 1;
        queue.push(q);
      }
    }
    return dist;
  });
}

/** Every bijection of n boxes to n goals, as permutations of the goal indices. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k: number): void => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      const tmp = a[k] as number;
      a[k] = a[i] as number;
      a[i] = tmp;
      gen(k + 1);
      a[i] = a[k] as number;
      a[k] = tmp;
    }
  };
  gen(0);
  return out;
}

/** Sorted boxes (at most four) packed into one number, base 1024, padded with the sentinel 1023. */
function packBoxes(boxes: number[]): number {
  let p = 0;
  for (let i = 0; i < 4; i++) p += (i < boxes.length ? (boxes[i] as number) : 1023) * (POW[i] as number);
  return p;
}

/** The inverse of packBoxes: the first n box squares, in ascending order. */
function unpackBoxes(p: number, n: number, out: number[]): void {
  out.length = 0;
  for (let i = 0; i < n; i++) out.push(Math.floor(p / (POW[i] as number)) % 1024);
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(wall: Uint8Array, nbr: Int32Array, occ: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi] as number;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = nbr[p * 4 + k] as number;
      if (q < 0 || wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = DIRS[k]?.[2] as string;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p] as number) s = (how[p] as string) + s;
  return s;
}

/**
 * One room, solved by A* over pushes (or, when greedy, by best-first search on
 * the estimate alone, which finds a solution fast but not necessarily the
 * cheapest). A state is the boxes plus the player's reachable region, named by
 * its lowest square; the estimate is the cheapest matching of boxes to goals
 * over walking distances, which one push can lower by at most one, so the
 * search is push-optimal. Dead squares and frozen 2 x 2 blocks are pruned.
 * Returns null when the work budget runs out or the room cannot be solved.
 */
function searchRoom(
  wall: Uint8Array,
  nbr: Int32Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  budget: number,
  greedy: boolean,
): RoomSolution | null {
  const size = wall.length;
  const n = boxes0.length;
  if (n === 0) return { pushes: 0, moves: '' };
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(wall, nbr, goals);
  const boxes = boxes0.slice().sort((a, b) => a - b);
  for (const b of boxes) if (!live[b]) return null;
  const dist = goalDistances(wall, nbr, goals);
  const perms = permutations(n);

  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  let seenStamp = 0;
  let markStamp = 0;
  const stack: number[] = [];

  // Flood-fill from `from` over open floor not holding a box, marking `arr`
  // with `stamp`; returns the region's lowest square.
  const flood = (from: number, arr: Int32Array, stamp: number): number => {
    let low = from;
    stack.length = 0;
    stack.push(from);
    arr[from] = stamp;
    while (stack.length > 0) {
      const p = stack.pop() as number;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = nbr[p * 4 + k] as number;
        if (q >= 0 && !wall[q] && !occ[q] && arr[q] !== stamp) {
          arr[q] = stamp;
          stack.push(q);
        }
      }
    }
    return low;
  };

  // Whether the box just pushed to `t` sits in a frozen 2 x 2 block: four
  // squares of walls and boxes, a box among them off its goal. Such a box can
  // never move again, since every push of it needs the player inside the block.
  const frozenAt = (t: number): boolean => {
    const up = nbr[t * 4] as number;
    const corners = [t, nbr[t * 4 + 2] as number, up, up >= 0 ? (nbr[up * 4 + 2] as number) : -1];
    for (const a of corners) {
      if (a < 0) continue;
      const r = nbr[a * 4 + 3] as number;
      const d = nbr[a * 4 + 1] as number;
      const dr = d >= 0 ? (nbr[d * 4 + 3] as number) : -1;
      let full = true;
      let offGoal = false;
      for (const c of [a, r, d, dr]) {
        if (c < 0 || wall[c] || occ[c]) {
          if (c >= 0 && occ[c] && !goalSet[c]) offGoal = true;
        } else {
          full = false;
          break;
        }
      }
      if (full && offGoal) return true;
    }
    return false;
  };

  // The cheapest matching of boxes to goals over walking distances; Infinity
  // when some box cannot walk to any goal at all.
  const heuristic = (bs: number[]): number => {
    let bestH = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = (dist[perm[i] as number] as Int32Array)[bs[i] as number] as number;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < bestH) bestH = sum;
    }
    return bestH;
  };

  // The states, in the order created: packed boxes, the player's square after
  // the push that made the state, the parent, that push (from * size + to),
  // pushes so far, the estimate, and the state's key.
  const sBoxes: number[] = [];
  const sPlayer: number[] = [];
  const sParent: number[] = [];
  const sPush: number[] = [];
  const sG: number[] = [];
  const sH: number[] = [];
  const sKey: number[] = [];
  const best = new Map<number, number>();

  // A binary heap of state ids, ordered by (priority, estimate, id): the
  // priority is pushes + estimate (A*) or the estimate alone (greedy).
  const heap: number[] = [];
  const priority = (id: number): number => (greedy ? (sH[id] as number) : (sG[id] as number) + (sH[id] as number));
  const heapLess = (a: number, b: number): boolean => {
    const pa = priority(a);
    const pb = priority(b);
    if (pa !== pb) return pa < pb;
    if (sH[a] !== sH[b]) return (sH[a] as number) < (sH[b] as number);
    return a < b;
  };
  const heapPush = (id: number): void => {
    heap.push(id);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i] as number, heap[p] as number)) break;
      const tmp = heap[i] as number;
      heap[i] = heap[p] as number;
      heap[p] = tmp;
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heap[0] as number;
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1] as number, heap[c] as number)) c++;
        if (!heapLess(heap[c] as number, heap[i] as number)) break;
        const tmp = heap[i] as number;
        heap[i] = heap[c] as number;
        heap[c] = tmp;
        i = c;
      }
    }
    return top;
  };

  // The moves of a solved state: walk to each push's square, then push.
  const reconstruct = (goalState: number): RoomSolution => {
    const pushes: number[] = [];
    for (let s = goalState; s > 0; s = sParent[s] as number) pushes.push(sPush[s] as number);
    pushes.reverse();
    let moves = '';
    let player = player0;
    const curBoxes = boxes0.slice().sort((a, b) => a - b);
    const occ2 = new Uint8Array(size);
    for (const b of curBoxes) occ2[b] = 1;
    for (const pk of pushes) {
      const from = Math.floor(pk / size);
      const to = pk % size;
      let k = 0;
      for (let j = 0; j < 4; j++) {
        if (nbr[from * 4 + j] === to) {
          k = j;
          break;
        }
      }
      const pushPos = nbr[from * 4 + OPP[k]] as number;
      moves += walkTo(wall, nbr, occ2, player, pushPos);
      moves += (DIRS[k]?.[2] as string).toUpperCase();
      occ2[from] = 0;
      occ2[to] = 1;
      const idx = curBoxes.indexOf(from);
      curBoxes[idx] = to;
      curBoxes.sort((a, b) => a - b);
      player = from;
    }
    return { pushes: pushes.length, moves };
  };

  // The starting state: a frozen block already in place means the room is lost.
  for (const b of boxes) occ[b] = 1;
  for (const b of boxes) if (frozenAt(b)) return null;
  const low0 = flood(player0, seen, ++seenStamp);
  const h0 = heuristic(boxes);
  if (h0 === Infinity) return null;
  const packed0 = packBoxes(boxes);
  const key0 = low0 * (POW[4] as number) + packed0;
  sBoxes.push(packed0);
  sPlayer.push(player0);
  sParent.push(-1);
  sPush.push(-1);
  sG.push(0);
  sH.push(h0);
  sKey.push(key0);
  best.set(key0, 0);
  heapPush(0);
  let created = 1;
  // The best solved state met so far (and its push count): if the work budget
  // runs out mid-search, this is still a solution — possibly a few pushes over
  // the optimum, but worth far more than giving up on the room.
  let bestGoal = -1;
  let bestGoalG = Infinity;

  const cur: number[] = [];
  const next: number[] = [];
  while (heap.length > 0) {
    const s = heapPop();
    const g = sG[s] as number;
    if (!greedy && (best.get(sKey[s] as number) as number) < g) continue; // a better path to this state exists
    unpackBoxes(sBoxes[s] as number, n, cur);
    if (cur.every((b) => goalSet[b])) return reconstruct(s);
    for (const b of cur) occ[b] = 1;
    const here = ++seenStamp;
    flood(sPlayer[s] as number, seen, here);
    for (let bi = 0; bi < n; bi++) {
      const b = cur[bi] as number;
      for (let k = 0; k < 4; k++) {
        const pushPos = nbr[b * 4 + OPP[k]] as number;
        const dest = nbr[b * 4 + k] as number;
        if (pushPos < 0 || dest < 0) continue;
        if (seen[pushPos] !== here) continue; // the player cannot reach the pushing square
        if (wall[dest] || occ[dest] || !live[dest]) continue;
        next.length = 0;
        for (let i = 0; i < n; i++) next.push(i === bi ? dest : (cur[i] as number));
        next.sort((a, b2) => a - b2);
        const packed = packBoxes(next);
        occ[b] = 0;
        occ[dest] = 1;
        const frozen = frozenAt(dest);
        let key = -1;
        if (!frozen) key = flood(b, mark, ++markStamp) * (POW[4] as number) + packed;
        occ[dest] = 0;
        occ[b] = 1;
        if (frozen) continue;
        const ng = g + 1;
        if (greedy) {
          if (best.has(key)) continue;
        } else {
          const prev = best.get(key);
          if (prev !== undefined && prev <= ng) continue;
        }
        const nh = heuristic(next);
        if (nh === Infinity) continue;
        if (created >= budget) return bestGoal >= 0 ? reconstruct(bestGoal) : null;
        best.set(key, greedy ? 0 : ng);
        const id = sBoxes.length;
        sBoxes.push(packed);
        sPlayer.push(b);
        sParent.push(s);
        sPush.push(b * size + dest);
        sG.push(ng);
        sH.push(nh);
        sKey.push(key);
        heapPush(id);
        created++;
        if (ng < bestGoalG && next.every((b2) => goalSet[b2])) {
          bestGoal = id;
          bestGoalG = ng;
        }
      }
    }
    for (const b of cur) occ[b] = 0;
  }
  return null;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor, and each part left that holds
 * a box is a room. Null when some part holds more or fewer boxes than goals,
 * or a box starts in the hallway: then the level is no set of rooms.
 */
function findRooms(level: Level, nbr: Int32Array): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(wall, nbr, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i] as number;
      for (let k = 0; k < 4; k++) {
        const q = nbr[p * 4 + k] as number;
        if (q >= 0 && !wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    (rooms[part[b]] as Room).boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    (rooms[part[g]] as Room).goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The square the player stands on after `moves` (LURD, the case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Replay the moves on the level, exactly as the checker does: every step legal, every box on a goal at the end. */
function replayOk(level: Level, moves: string): boolean {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  if (moves.length > 65536) return false;
  const box = new Uint8Array(size);
  for (const b of boxes) box[b] = 1;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  let p = player;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    if (c !== 'u' && c !== 'd' && c !== 'l' && c !== 'r') return false;
    const d = c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
    const next = p + d;
    if (next < 0 || next >= size || wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (beyond < 0 || beyond >= size || wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    p = next;
  }
  for (let i = 0; i < size; i++) if (box[i] && !goal[i]) return false;
  return true;
}

/** One level's moves, or the empty string when no solution was found in budget. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, wall, goals, boxes, player } = level;
  if (player < 0 || boxes.length === 0 || w * level.h > 1023) return '';
  const nbr = neighborTable(w, level.h);
  const rooms = findRooms(level, nbr);
  if (rooms === null) {
    // No clean room split: try the whole level as one room.
    const res =
      searchRoom(wall, nbr, goals, boxes, player, ASTAR_BUDGET, false) ??
      searchRoom(wall, nbr, goals, boxes, player, GREEDY_BUDGET, true);
    return res !== null && replayOk(level, res.moves) ? res.moves : '';
  }
  // Room by room, in scan order; the other rooms' boxes stand as walls, on
  // their goals once their room is done. The player walks on from where the
  // last room left it, through the hallway, into the next room.
  const roomBoxes = rooms.map((r) => r.boxes.slice().sort((a, b) => a - b));
  let moves = '';
  let curPlayer = player;
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri] as Room;
    const subWall = wall.slice();
    for (let rj = 0; rj < rooms.length; rj++) {
      if (rj === ri) continue;
      for (const b of roomBoxes[rj] as number[]) subWall[b] = 1;
    }
    const res =
      searchRoom(subWall, nbr, room.goals, roomBoxes[ri] as number[], curPlayer, ASTAR_BUDGET, false) ??
      searchRoom(subWall, nbr, room.goals, roomBoxes[ri] as number[], curPlayer, GREEDY_BUDGET, true);
    if (res === null) return '';
    moves += res.moves;
    roomBoxes[ri] = room.goals.slice().sort((a, b) => a - b);
    curPlayer = endOf(w, curPlayer, res.moves);
  }
  return replayOk(level, moves) ? moves : '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch (err) {
    console.error(`sokoban-solver: ${id}: ${(err as Error).message}`);
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}

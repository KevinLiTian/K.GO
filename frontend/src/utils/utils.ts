function contains(array: number[][], tuple: [number, number]) {
  for (const element of array) {
    if (tuple[0] === element[0] && tuple[1] === element[1]) {
      return true;
    }
  }
  return false;
}

export function isStar(row: number, col: number) {
  let stars = [];
  stars.push([3, 3]);
  stars.push([3, 9]);
  stars.push([3, 15]);
  stars.push([9, 3]);
  stars.push([9, 15]);
  stars.push([15, 3]);
  stars.push([15, 9]);
  stars.push([15, 15]);
  stars.push([9, 9]);

  return contains(stars, [row, col]);
}

export function playerColor(player: number) {
  return player === 1 ? 'black' : 'white';
}

export function copyBoard(board: number[][]) {
  return JSON.parse(JSON.stringify(board));
}

export function getBoardHash(board: number[][]) {
  let hash = 0;
  for (let i = 0; i < 19; i++) {
    for (let j = 0; j < 19; j++) {
      hash = (hash * 3 + board[i][j]) % 1000000;
    }
  }
  return hash;
}

export function createBoard(): number[][] {
  let outer = [];
  for (let i = 0; i < 19; i++) {
    let inner = [];
    for (let j = 0; j < 19; j++) {
      inner.push(0);
    }
    outer.push(inner);
  }
  return outer;
}

function createVisited(): boolean[][] {
  let outer = [];
  for (let i = 0; i < 19; i++) {
    let inner = [];
    for (let j = 0; j < 19; j++) {
      inner.push(false);
    }
    outer.push(inner);
  }
  return outer;
}

function findGroups(board: number[][]) {
  const groups: number[][][] = [];
  const visited: boolean[][] = createVisited();

  // Iterate over every cell on the board
  for (let row = 0; row < 19; row++) {
    for (let col = 0; col < 19; col++) {
      // If not empty and not visited, visit it
      if (board[row][col] !== 0 && !visited[row][col]) {
        visited[row][col] = true;
        const group: number[][] = [[row, col]];

        // DFS
        const frontier: number[][] = [[row, col]];
        while (frontier.length !== 0) {
          const [x, y] = frontier.pop()!;
          for (const [r, c] of findAdjacentCells(x, y)) {
            // Same colour, within board and not visited
            if (
              withinBoard(r, c) &&
              board[r][c] === board[x][y] &&
              !visited[r][c]
            ) {
              group.push([r, c]);
              visited[r][c] = true;
              frontier.push([r, c]);
            }
          }
        }
        groups.push(group);
      }
    }
  }

  return groups;
}

function findAdjacentCells(row: number, col: number) {
  const adjacent: number[][] = [];
  [
    [0, 1],
    [0, -1],
    [1, 0],
    [-1, 0],
  ].forEach(([dx, dy]) => {
    if (withinBoard(row + dx, col + dy)) {
      adjacent.push([row + dx, col + dy]);
    }
  });

  return adjacent;
}

function withinBoard(row: number, col: number) {
  return row >= 0 && row < 19 && col >= 0 && col < 19;
}

function findGroupLiberties(board: number[][], group: number[][]) {
  let liberties = 0;
  const visited: boolean[][] = createVisited();

  // Find liberties of each cell (consider overlap liberties)
  for (const [row, col] of group) {
    for (const [x, y] of findAdjacentCells(row, col)) {
      if (board[x][y] === 0 && !visited[x][y]) {
        visited[x][y] = true;
        liberties += 1;
      }
    }
  }

  return liberties;
}

function findStoneGroup(row: number, col: number, groups: number[][][]) {
  for (let i = 0; i < groups.length; i++) {
    if (contains(groups[i], [row, col])) {
      return i;
    }
  }
}

export function getGroups(
  board: number[][],
  row: number,
  col: number
): [number[][][], number, number[]] {
  const groups = findGroups(board);
  const curStoneGroup = findStoneGroup(row, col, groups)!;
  const deadGroups: number[] = [];

  for (let i = 0; i < groups.length; i++) {
    if (findGroupLiberties(board, groups[i]) === 0) {
      deadGroups.push(i);
    }
  }

  return [groups, curStoneGroup, deadGroups];
}

export function removeDeadGroups(
  board: number[][],
  groups: number[][][],
  curStoneGroup: number,
  deadGroups: number[]
) {
  for (let i of deadGroups) {
    if (i !== curStoneGroup) {
      for (let [row, col] of groups[i]) {
        board[row][col] = 0;
      }
    }
  }
}

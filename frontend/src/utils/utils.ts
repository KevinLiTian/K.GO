function contains(array: number[][], tuple: [number, number]) {
  for (const element of array) {
    if (tuple[0] === element[0] && tuple[1] === element[1]) {
      return true;
    }
  }
  return false;
}

function getBoardHash(board: number[][]) {
  let hash = 0;
  for (let i = 0; i < 19; i++) {
    for (let j = 0; j < 19; j++) {
      hash = (hash * 3 + board[i][j]) % 1000000;
    }
  }
  return hash;
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

export function playerColor(player: number) {
  return player === 1 ? 'black' : 'white';
}

export function copyBoard(board: number[][]) {
  return JSON.parse(JSON.stringify(board));
}

export function isKO(board: number[][], previousBoards: number[]) {
  const hash = getBoardHash(board);
  if (previousBoards.includes(hash)) {
    return true;
  }
  previousBoards.push(hash);
  if (previousBoards.length > 8) {
    previousBoards.shift();
  }
  return false;
}

function findGroups(board: number[][]) {
  let groups: number[][][] = [];
  let visited: boolean[][] = createVisited();

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
          const pos = frontier.pop()!;
          for (const cell of findAdjacentCells(pos[0], pos[1])) {
            // Same colour, within board and not visited
            if (
              withinBoard(cell[0], cell[1]) &&
              board[cell[0]][cell[1]] === board[pos[0]][pos[1]] &&
              !visited[cell[0]][cell[1]]
            ) {
              group.push(cell);
              visited[cell[0]][cell[1]] = true;
              frontier.push(cell);
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
  for (let i = row - 1; i <= row + 1; i++) {
    for (let j = col - 1; j <= col + 1; j++) {
      if (withinBoard(i, j)) {
        adjacent.push([i, j]);
      }
    }
  }

  return adjacent;
}

function withinBoard(row: number, col: number) {
  return row >= 0 && row < 19 && col >= 0 && col < 19;
}

export function updateBoard(board: number[][]) {
  const groups = findGroups(board);
}

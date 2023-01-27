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

  return contains(stars, [row, col]);
}

export function createBoard(): number[][] {
  return Array(19).fill(Array(19).fill(0));
}

export function playerColor(player: number) {
  return player === 1 ? 'black' : 'white';
}

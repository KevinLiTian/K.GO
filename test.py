import unittest

from Go.Go import Go
from preprocessing import create_board_state


class TestKo(unittest.TestCase):
    def test_standard_ko(self):

        gs = Go()

        gs.make_move(1, 0)  # B
        gs.make_move(2, 0)  # W
        gs.make_move(0, 1)  # B
        gs.make_move(3, 1)  # W
        gs.make_move(1, 2)  # B
        gs.make_move(2, 2)  # W
        gs.make_move(2, 1)  # B

        gs.make_move(1, 1)  # W trigger capture and ko

        self.assertFalse(gs.is_legal(2, 1))

        gs.make_move(5, 5)
        gs.make_move(5, 6)

        self.assertTrue(gs.is_legal(2, 1))

    def test_snapback_is_not_ko(self):

        gs = Go()

        # B o W B .
        # W W B . .
        # . . . . .
        # . . . . .
        # . . . . .
        # here, imagine black plays at 'o' capturing
        # the white stone at (2, 0). White may play
        # again at (2, 0) to capture the black stones
        # at (0, 0), (1, 0). this is 'snapback' not 'ko'
        # since it doesn't return the game to a
        # previous position

        B = [(0, 0), (2, 1), (3, 0)]
        W = [(0, 1), (1, 1), (2, 0)]
        for (b, w) in zip(B, W):
            gs.make_move(b[0], b[1])
            gs.make_move(w[0], w[1])

        # do the capture of the single white stone
        gs.make_move(1, 0)

        # there should be no ko
        self.assertIsNone(gs.ko)
        self.assertTrue(gs.is_legal(2, 0))

        # now play the snapback
        gs.make_move(2, 0)

    def test_positional_superko(self):

        # test with enforce_superko=False
        gs = Go()

        move_list = [
            (0, 3),
            (0, 4),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (2, 2),
            (3, 4),
            (2, 1),
            (3, 3),
            (3, 1),
            (3, 2),
            (3, 0),
            (4, 2),
            (1, 1),
            (4, 1),
            (8, 0),
            (4, 0),
            (8, 1),
            (0, 2),
            (8, 2),
            (0, 1),
            (8, 3),
            (1, 0),
            (8, 4),
            (2, 0),
            (0, 0),
        ]

        for move in move_list:
            gs.make_move(move[0], move[1])

        self.assertTrue(gs.is_legal(1, 0))


class TestLadder(unittest.TestCase):
    def test_ladder_capture_escape(self):
        go = Go()

        go.make_move(3, 3)
        go.make_move(3, 4)
        go.make_move(2, 4)
        go.make_move(0, 0)
        go.make_move(4, 3)
        go.make_move(0, 1)

        # Is a capture
        self.assertEqual(go.get_ladder_capture()[0, 3, 5], 1)

        go.make_move(0, 2)

        # Not an escape
        self.assertEqual(go.get_ladder_escape()[0, 3, 5], 0)

        # 引征
        go.make_move(15, 15)

        # Not a capture anymore
        self.assertEqual(go.get_ladder_capture()[0, 3, 5], 0)

        # Start capture
        go.make_move(3, 5)

        # Now is an escape
        self.assertEqual(go.get_ladder_escape()[0, 4, 4], 1)


class TestEye(unittest.TestCase):
    def test_eye(self):
        go = Go()
        go.make_move(0, 1)
        go.make_move(0, 10)
        go.make_move(1, 0)

        self.assertFalse(go.is_eye((0, 0), 1))

        go.make_move(0, 12)
        go.make_move(1, 1)

        self.assertTrue(go.is_eye((0, 0), 1))

        go.make_move(1, 11)

        self.assertFalse(go.is_eye((0, 11), 2))

        go.make_move(9, 9)
        go.make_move(1, 10)

        self.assertFalse(go.is_eye((0, 11), 2))

        go.make_move(9, 10)
        go.make_move(1, 12)

        self.assertTrue(go.is_eye((0, 11), 2))

class TestBoardState(unittest.TestCase):
    def test_stone_colours(self):
        go = Go()

        go.make_move(3, 3)
        state = create_board_state(go)
        self.assertEqual(state[1, 3, 3], 1)
        
        go.make_move(15, 15)
        state = create_board_state(go)
        self.assertEqual(state[0, 3, 3], 1)
        self.assertEqual(state[1, 15, 15], 1)

        # Not empty
        self.assertEqual(state[2, 3, 3], 0)
        self.assertEqual(state[2, 15, 15], 0)

    def test_turns_since(self):
        go = Go()

        go.make_move(3, 3)
        go.make_move(15, 15)

        state = create_board_state(go)
        self.assertEqual(state[4, 15, 15], 1)
        self.assertEqual(state[5, 3, 3], 1)

    def test_liberties(self):
        go = Go()

        go.make_move(3, 3)
        state = create_board_state(go)
        self.assertEqual(state[15, 3, 3], 1)

        go.make_move(3, 4)
        state = create_board_state(go)
        self.assertEqual(state[14, 3, 3], 1)
        self.assertEqual(state[14, 3, 4], 1)

if __name__ == "__main__":
    unittest.main()

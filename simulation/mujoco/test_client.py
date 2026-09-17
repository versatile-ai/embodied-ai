"""Policy boundary regression tests; no network or physics required."""
import unittest
import copy
from unittest.mock import patch
import numpy as np
import client


class ActionBoundary(unittest.TestCase):
    def test_clips_only_grippers_and_preserves_proposal(self):
        actions = np.zeros((2, 14))
        actions[:, 6] = -0.02
        actions[:, 13] = 1.03
        actions[:, 0] = 0.5
        original = actions.copy()
        payload = {'joints': actions.tolist()}
        before = copy.deepcopy(payload)
        with patch.object(client, 'http', return_value={}) as call:
            client.execute('test', 0, 'act', payload, 'follow')
        body = call.call_args.args[1]
        result = np.asarray(body['joints'])
        np.testing.assert_array_equal(actions, original)
        self.assertEqual(payload, before)
        np.testing.assert_array_equal(result[:, 6], 0)
        np.testing.assert_array_equal(result[:, 13], 1)
        np.testing.assert_array_equal(result[:, 0], 0.5)
        self.assertIn('clipped to [0,1]: 4', body['decision_summary'])

    def test_rejects_nonfinite_without_sending(self):
        actions = [[0.] * 14]
        actions[0][6] = float('nan')
        with patch.object(client, 'http') as call:
            with self.assertRaises(ValueError):
                client.execute('test', 0, 'act', {'joints': actions}, 'follow')
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)

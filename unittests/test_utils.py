import unittest
from unittest.mock import patch, MagicMock
import torch
import logging
import torch
import torch.nn as nn

# Import the function from your module, e.g.:
# from ml_module.config.paths import setup_environment_and_devices
from Multimodal_AUV.config.paths import setup_environment_and_devices
from Multimodal_AUV.utils.device import move_model_to_device, move_models_to_device, check_model_devices

class TestSetupEnvironmentAndDevices(unittest.TestCase):

    @patch('Multimodal_AUV.config.paths.get_environment_paths')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_returns_cpu_device_when_cuda_not_available(self, mock_device_count, mock_is_available, mock_get_paths):
        # Mock environment paths
        mock_get_paths.return_value = ('/root', '/models', '/strangford', '/mulroy')
        mock_is_available.return_value = False

        root_dir, models_dir, strangford_dir, mulroy_dir, devices = setup_environment_and_devices()

        self.assertEqual(root_dir, '/root')
        self.assertEqual(models_dir, '/models')
        self.assertEqual(strangford_dir, '/strangford')
        self.assertEqual(mulroy_dir, '/mulroy')

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0], torch.device('cpu') or torch.device('cuda'))

    @patch('Multimodal_AUV.config.paths.get_environment_paths')
    @patch('torch.cuda.is_available')
    def test_force_cpu_overrides_cuda(self, mock_is_available, mock_get_paths):
        mock_get_paths.return_value = ('/root', '/models', '/strangford', '/mulroy')
        mock_is_available.return_value = True

        root_dir, models_dir, strangford_dir, mulroy_dir, devices = setup_environment_and_devices(force_cpu=True)

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0], torch.device('cpu'))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

class TestDeviceUtils(unittest.TestCase):

    def setUp(self):
        # Dummy model instance for tests
        self.model = DummyModel()

    @patch('torch.nn.DataParallel')
    def test_move_model_to_device_single(self, mock_dataparallel):
        device = torch.device('cpu')
        # No multiple GPUs, so DataParallel should NOT be called
        moved_model = move_model_to_device(self.model, device, device_ids=None)
        self.assertEqual(moved_model, self.model.to(device))
        mock_dataparallel.assert_not_called()

    @patch('torch.nn.DataParallel')
    def test_move_model_to_device_multiple_gpus(self, mock_dataparallel):
        device = torch.device('cuda:0')
        mock_dataparallel.return_value = 'wrapped_model'
        if torch.cuda.device_count() > 1:
            device_ids = [0, 1]
            moved_model = move_model_to_device(self.model, device, device_ids=device_ids)
            # Should wrap model in DataParallel and return it
            mock_dataparallel.assert_called_once_with(self.model.to(device), device_ids=device_ids)
            self.assertEqual(moved_model, 'wrapped_model')

    def test_move_model_to_device_exception(self):
        # Make model.to raise an exception to test error handling
        class BadModel:
            def to(self, device):
                raise RuntimeError("fail")
        bad_model = BadModel()

        with self.assertRaises(RuntimeError):
            move_model_to_device(bad_model, torch.device('cpu'))

    @patch('torch.cuda.device_count', return_value=2)
    @patch('Multimodal_AUV.utils.device.move_model_to_device')
    def test_move_models_to_device(self, mock_move_model, mock_device_count):
        mock_move_model.side_effect = lambda m, d, device_ids=None: f"{m}_moved"

        models_dict = {
            "image_model": "image_model",
            "multimodal_model": "multimodal_model",
            "channels_model": None
        }
        devices = [torch.device('cuda:0'), torch.device('cuda:1')]

        moved = move_models_to_device(models_dict, devices, use_multigpu_for_multimodal=True)

        print("Calls to move_model_to_device:", mock_move_model.mock_calls)

        calls = mock_move_model.mock_calls

        found_multimodal_call = any(
            call[1][0] == "multimodal_model" and
            str(call[1][1]) == str(devices[0]) and
            len(call[1]) > 2 and
            call[1][2] == [0, 1]
            for call in calls
        )
        self.assertTrue(found_multimodal_call, "move_model_to_device not called correctly for multimodal_model with device_ids=[0,1]")

        found_image_model_call = any(
            call[1][0] == "image_model" and
            str(call[1][1]) == str(devices[0]) and
            (len(call[1]) < 3 or call[1][2] is None)
            for call in calls
        )
        self.assertTrue(found_image_model_call, "move_model_to_device not called correctly for image_model without device_ids")

        self.assertIsNone(moved["channels_model"])
        self.assertEqual(moved["image_model"], "image_model_moved")
        self.assertEqual(moved["multimodal_model"], "multimodal_model_moved")

    def test_check_model_devices_all_correct(self):
        device = torch.device('cpu')
        model = DummyModel()
        model.to(device)
        with self.assertLogs(level='INFO') as log:
            result = check_model_devices(model, device)
        self.assertTrue(result)
        self.assertIn("All model parameters are on the expected device.", log.output[0])

    def test_check_model_devices_some_wrong(self):
        # Param on cpu but expected cuda, so should warn and return False
        model = DummyModel()
        model.to(torch.device('cpu'))
        expected_device = torch.device('cuda:0')
        with self.assertLogs(level='WARNING') as log:
            result = check_model_devices(model, expected_device)
        self.assertFalse(result)
        self.assertIn("Param linear.weight is on cpu, expected cuda:0", log.output[0])

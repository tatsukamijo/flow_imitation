import torch
from flow_imitation.common.policies.flow.modeling_flow import FlowPolicy, FlowConfig


def make_dummy_config():
    class DummyFeature:
        shape = (4,)

    class DummyImageFeature:
        shape = (3, 32, 32)

    return FlowConfig(
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        input_features={
            "observation.state": DummyFeature(),
            "observation.images": DummyImageFeature(),
        },
        output_features={"action": DummyFeature()},
        action_feature=DummyFeature(),
        robot_state_feature=DummyFeature(),
        env_state_feature=None,
        crop_shape=(32, 32),
    )


def make_dummy_batch(config):
    B = 2
    batch = {
        "observation.state": torch.randn(B, config.n_obs_steps, 4),
        "observation.images": torch.randn(B, config.n_obs_steps, 1, 3, 32, 32),
        "action": torch.randn(B, config.horizon, 4),
    }
    return batch


def test_flow_policy_forward():
    config = make_dummy_config()
    policy = FlowPolicy(config)
    batch = make_dummy_batch(config)
    loss, _ = policy.forward(batch)
    assert loss is not None and loss.item() >= 0


def test_flow_policy_select_action():
    config = make_dummy_config()
    policy = FlowPolicy(config)
    batch = make_dummy_batch(config)
    action = policy.select_action(
        {
            "observation.state": batch["observation.state"][0],
            "observation.images": batch["observation.images"][0],
        }
    )
    assert action.shape[-1] == 4


def test_flow_model_generate_actions():
    config = make_dummy_config()
    policy = FlowPolicy(config)
    batch = make_dummy_batch(config)
    actions = policy.flow.generate_actions(batch)
    assert actions.shape[0] == 2

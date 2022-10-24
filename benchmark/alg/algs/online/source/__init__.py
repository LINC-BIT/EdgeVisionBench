from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker

from schema import Schema


@algorithm_register(
    name='Source',
    stage='online',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class Source(ABOnlineDAAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()
        assert len(models['main model']) > 0, 'pass the path of model file in'
        
        Schema({}).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        pass
    
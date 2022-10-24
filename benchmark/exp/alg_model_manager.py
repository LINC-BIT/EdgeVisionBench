from abc import ABC, abstractmethod
import copy


class ABAlgModelsManager(ABC):
    @abstractmethod
    def forward_to_compute_loss(self, models, x, y):
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, models, x):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, models, x):
        raise NotImplementedError
    
    @abstractmethod
    def get_accuracy(self, models, test_dataloader):
        raise NotImplementedError

    def get_model(self, models, key):
        model = models[key]
        if isinstance(model, tuple):
            model = model[0]
        return model
    
    def set_model(self, models, key, model):
        if key in models.keys() and isinstance(models[key], tuple):
            models[key] = (model, *models[key])
        else:
            models[key] = model
    
    def get_model_desc(self, models):
        desc = []
        for key, model_info in models.items():
            if isinstance(model_info, tuple):
                desc += [key + ' (' + ' / '.join(model_info[1:]) + ')']
            else:
                desc += [key]
        return '\n'.join(desc)

    def get_deepcopied_models(self, models):
        res = {}
        def try_deepcopy(model):
            try:
                return copy.deepcopy(model)
            except Exception as e:
                print('deepcopy exception: ' + str(e))
                return model
            
        for key, model_info in models.items():
            res[key] = try_deepcopy(model_info) if not isinstance(model_info, tuple) else (try_deepcopy(model_info[0]), *model_info[1:])
        return res

    def to_device(self, models, device):
        res = {}
        def try_to_device(model):
            try:
                return model.to(device)
            except Exception as e:
                print('to device exception: ' + str(e))
                return model
            
        for key, model_info in models.items():
            res[key] = try_to_device(device) if not isinstance(model_info, tuple) else (try_to_device(model_info[0]), *model_info[1:])
        return res

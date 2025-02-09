from common.data import Model, Implementation, Task
from common.data.model import Setup


class FullyJoinedSetupView(Setup):
    implementation: Implementation
    task: Task

class FullyJoinedModelView(Model):
    setup: FullyJoinedSetupView

# coding=utf-8
# Copyright 2021 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines all the tasks supported. Used to define enums in train_eval, etc."""

IBC_TASKS = ['REACH', 'PUSH', 'INSERT', 'PARTICLE', 'PUSH_DISCONTINUOUS',
             'PUSH_MULTIMODAL']
ADROIT_TASKS = ['pen-human-v0', 'hammer-human-v0', 'door-human-v0',
                'relocate-human-v0',]
D4RL_TASKS = ['antmaze-large-diverse-v0',
              'antmaze-large-play-v0',
              'antmaze-medium-diverse-v0',
              'antmaze-medium-play-v0',
              'halfcheetah-expert-v0',
              'halfcheetah-medium-expert-v0',
              'halfcheetah-medium-replay-v0',
              'halfcheetah-medium-v0',
              'hopper-expert-v0',
              'hopper-medium-expert-v0',
              'hopper-medium-replay-v0',
              'hopper-medium-v0',
              'kitchen-complete-v0',
              'kitchen-mixed-v0',
              'kitchen-partial-v0',
              'walker2d-expert-v0',
              'walker2d-medium-expert-v0',
              'walker2d-medium-replay-v0',
              'walker2d-medium-v0'] + ADROIT_TASKS

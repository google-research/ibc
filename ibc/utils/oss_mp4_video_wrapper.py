# coding=utf-8
# Copyright 2022 The Reach ML Authors.
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

"""Open-source video wrapper based on OpenCV."""

import os
import cv2


class Mp4VideoWrapper():
  """Open-source environment wrapper that adds frames to an mp4 video."""

  def __init__(self,
               env,
               frame_rate,
               frame_interval,
               video_filepath):
    self._video_recorder = OssMp4VideoRecorder(video_filepath, frame_rate)
    self.batched = False
    self.env = env
    self.env.episode_steps = 0
    self._frame_interval = frame_interval

  def reset(self):
    time_step = self.env.reset()
    self.env.episode_steps = 0
    self._add_frame()
    return time_step

  def step(self, action):
    time_step = self.env.step(action)
    if not self.env.episode_steps % self._frame_interval:
      self._add_frame()
    self.env.episode_steps += 1
    return time_step

  def close(self):
    if self._video_recorder is None:
      raise ValueError("Already ended this video! I'm a one-time-use wrapper.")
    self._video_recorder.end_video()
    self._video_recorder = None

  def _add_frame(self):
    frame = self.env.render()
    self._video_recorder.add_frame(frame)


class OssMp4VideoRecorder():
  """Open-source Mp4VideoRecorder for creating mp4 videos frame by frame."""

  def __init__(self, filepath, frame_rate):
    self.filepath = filepath
    self.frame_rate = frame_rate
    self.vid_writer = None
    basedir = os.path.dirname(self.filepath)
    if not os.path.isdir(basedir):
      os.system("mkdir -p " + basedir)
    self.last_frame = None  # buffer so we don't write the last one.

  def init_vid_writer(self, width, height):
    self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    self.vid_writer = cv2.VideoWriter(self.filepath, self.fourcc,
                                      self.frame_rate, (width, height))

  def add_frame(self, frame):
    """Adds a frame to the video recorder.

    Args:
      frame: numpy array of shape [height, width, 3] representing the frame
        to add to the video.
    """
    # make even to avoid codec issues
    (h, w, _) = frame.shape
    if (h % 2 != 0) or (w % 2 != 0):
      if h % 2 != 0:
        h -= 1
      if w % 2 != 0:
        w -= 1
      frame = frame[:h, :w, :]

    if self.vid_writer is None:
      self.init_vid_writer(w, h)

    # :facepalm: why did opencv ever choose BGR?
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if self.last_frame is not None:
      self.vid_writer.write(self.last_frame)
    self.last_frame = frame

  def end_video(self):
    """Closes the video recorder and writes the frame buffer to disk."""
    self.vid_writer.release()

_base_ = [
    '../../_base_/datasets/coronary_video/coronary-2_image.py',
    '../../_base_/default_runtime/default_runtime_e.py',
    '../../_base_/models/semi3dtenet/semi3dtenet_fcn.py', '../../_base_/schedules/schedule_e.py'
]
# datasets
data = dict(samples_per_gpu=4, workers_per_gpu=4,
            train=dict(times=4, dataset=dict(frame_num=3)),
            val=dict(frame_num=3), test=dict(frame_num=3))
# default_runtime
log_config = dict(interval=166)
# models
# schedules
optimizer = dict(type='SGD', lr=0.01)
runner = dict(max_epochs=100)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mFscore', save_best='mFscore', rule='greater')

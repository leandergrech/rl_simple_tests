from gym.envs.registration import register

register(id='simpleEnv-v0',
         entry_point='simpleEnv.envs:simpleEnv'
         )

register(id='MORsimpleEnv-v0',
         entry_point='simpleEnv.envs:MORsimpleEnv'
         )

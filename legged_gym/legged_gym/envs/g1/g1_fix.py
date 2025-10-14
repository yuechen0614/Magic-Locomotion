from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1FixCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.78]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.
        }
         
    class sim(LeggedRobotCfg.sim):
        class physx( LeggedRobotCfg.sim.physx ):
            # num_threads = 12
            # max_gpu_contact_pairs = 2**25 #2**24 -> needed for 8000 envs and more
            # default_buffer_size_multiplier = 8
            pass

    class env( LeggedRobotCfg.env ):
        num_envs = 2048
        n_scan = 132
        n_priv = 3 + 3 + 3 # = 9 base velocity 3个
        # n_priv_latent = 4 + 1 + 12 +12
        n_priv_latent = 4 + 1 + 12 + 12 # mass, fraction, motor strength1 and 2
        
        n_proprio = 51 # 所有本体感知信息，即obs_buf
        history_len = 10

        student_obs = 46 + n_scan

        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv + student_obs#n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 

        contact_buf_len = 100
        episode_length_s = 40

    class depth( LeggedRobotCfg.depth ):
        position = [0.1, 0, 0.77]  # front camera
        angle = [-5, 5]  # positive pitch down
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof_with_hand.urdf'
        name = "g1_fix_upper"
        foot_name = "ankle_roll"
        knee_name = "knee"
        hip_name = "hip"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [0.1, 0.6]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values

  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        max_contact_force = 600
        is_play = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # tracking_lin_vel = 5.0
            # tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1e-2
            base_height = -10.0
            dof_acc = -2.5e-7
            action_rate = -0.02
            dof_pos_limits = -5.0

            feet_yaw = -5e-2
            # knee_yaw = -6e-1
            # feet_dis = -5e-4
            # knee_dis = -2e-2
            feet_height = 2.5

            tracking_goal_vel = 4
            tracking_yaw = 2
            # feet_parallel = -3.0
            smoothness = -0.01
            collision = -5
            # termination = -0.02
            alive = 0.1
            # feet_contact_forces = -0.01
            # fallen = -200
            self_roll = -6e-3
            self_yaw = -2e-4

class G1FixCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'Distillation'
        run_name = ''
        experiment_name = 'g1_fix'
        max_iterations = 50001 # number of policy updates
        save_interval = 200
        num_steps_per_env = 24

    class estimator(LeggedRobotCfgPPO.estimator):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = G1FixCfg.env.n_priv
        num_prop = G1FixCfg.env.n_proprio
        num_scan = G1FixCfg.env.n_scan
# OLD_AWD - Rewards, Pénalités et Observations

Ce document récapitule tous les rewards, pénalités et observations extraits du code original dans old_awd.

## 1. OBSERVATIONS

### 1.1 Duckling Base (duckling.py)

#### Observations de base (51 dimensions):
```python
# Ligne 1280-1288 de duckling.py
obs = torch.cat(
    (
        projected_gravity,      # 3 dims
        dof_pos,               # 16 dims
        dof_vel,               # 16 dims
        actions,               # 16 dims (prev_actions)
    ),
    dim=-1,
)
```

**Total: 51 dimensions**

#### Détails des composants:
- **projected_gravity** (3 dims): Gravité projetée dans le référentiel local du robot
- **dof_pos** (16 dims): Positions des articulations
- **dof_vel** (16 dims): Vitesses des articulations
- **actions** (16 dims): Actions précédentes

### 1.2 DucklingCommand (duckling_command.py)

#### Observations additionnelles (3 dimensions):
```python
# Ligne 182-184
obs = self.commands * self.commands_scale
```

**Observations spécifiques à la tâche:**
- **commands[0]** × lin_vel_scale: Commande de vitesse linéaire en X
- **commands[1]** × lin_vel_scale: Commande de vitesse linéaire en Y
- **commands[2]** × ang_vel_scale: Commande de vitesse angulaire en Z

**Total DucklingCommand: 54 dimensions** (51 base + 3 task)

### 1.3 DucklingHeading (duckling_heading.py)

#### Observations additionnelles (5 dimensions):
```python
# Ligne 341
obs = torch.cat([local_tar_dir, tar_speed, local_tar_face_dir], dim=-1)
```

**Observations spécifiques à la tâche:**
- **local_tar_dir** (2 dims): Direction cible dans le référentiel local
- **tar_speed** (1 dim): Vitesse cible
- **local_tar_face_dir** (2 dims): Direction de face cible dans le référentiel local

**Total DucklingHeading: 56 dimensions** (51 base + 5 task)

### 1.4 AMP Observations (duckling_amp.py)

#### Observations AMP (pour le discriminateur):
```python
# Ligne 474-485
obs = torch.cat(
    (
        root_h_obs,            # 1 dim (hauteur racine)
        root_rot_obs,          # 6 dims (rotation en tan_norm)
        local_root_vel,        # 3 dims
        local_root_ang_vel,    # 3 dims
        dof_obs,              # 96 dims (16 joints × 6)
        dof_vel,              # 16 dims
        flat_local_key_pos,   # 6 dims (2 key bodies × 3)
    ),
    dim=-1,
)
```

**Total AMP obs par step: 131 dimensions**
**Avec historique (5 steps): 655 dimensions**

## 2. REWARDS ET PÉNALITÉS

### 2.1 DucklingCommand Rewards

#### Configuration (duckling_command.yaml):
```yaml
linearVelocityXYRewardScale: 0.5
angularVelocityZRewardScale: 0.25
torqueRewardScale: -0.000025
actionRateRewardScale: 0.0
standStillRewardScale: 0.0  # Not defined in yaml, likely 0
```

#### Calcul des rewards (compute_task_reward):

1. **Reward de suivi de vitesse linéaire XY**
```python
lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * rew_scales["lin_vel_xy"]
```
- Récompense exponentielle basée sur l'erreur au carré
- Diviseur: 0.25
- Scale: 0.5

2. **Reward de suivi de vitesse angulaire Z**
```python
ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * rew_scales["ang_vel_z"]
```
- Récompense exponentielle basée sur l'erreur au carré
- Diviseur: 0.25
- Scale: 0.25

3. **Pénalité de couple (torque)**
```python
rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
```
- Pénalité quadratique sur les couples
- Scale: -0.000025 × dt

4. **Pénalité de variation d'action**
```python
rew_action_rate = torch.sum(torch.square(prev_actions - actions), dim=1) * rew_scales["action_rate"]
```
- Pénalité quadratique sur la différence entre actions consécutives
- Scale: 0.0 × dt (désactivé)

5. **Pénalité d'immobilité**
```python
rew_stand_still = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1) *
                  (torch.norm(commands, dim=1) < 0.01) * rew_scales["stand_still"]
```
- Active seulement quand la norme des commandes < 0.01
- Scale: 0.0 (désactivé)

**Reward total:**
```python
total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_action_rate + rew_stand_still
total_reward = torch.clip(total_reward, 0.0, None)  # Clamp à 0 minimum
```

### 2.2 DucklingHeading Rewards

#### Calcul (compute_heading_reward):

1. **Direction reward (70% du total)**
```python
vel_err_scale = 0.25
tangent_err_w = 0.1
dir_reward_w = 0.7

tar_vel_err = tar_speed - tar_dir_speed
tangent_vel_err = tangent_speed
dir_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err +
                       tangent_err_w * tangent_vel_err * tangent_vel_err))
```
- Récompense pour suivre la vitesse cible dans la bonne direction
- Pénalité pour vitesse tangentielle (pondération: 0.1)
- Mis à 0 si vitesse dans mauvaise direction (tar_dir_speed <= 0)

2. **Facing reward (30% du total)**
```python
facing_reward_w = 0.3
facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
facing_reward = torch.clamp_min(facing_err, 0.0)
```
- Récompense pour orienter le robot dans la bonne direction
- Produit scalaire entre direction actuelle et cible
- Clampé à 0 minimum

**Reward total:**
```python
reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward
```

### 2.3 Duckling Base Reward

```python
# Ligne 1425-1428
def compute_duckling_reward(obs_buf):
    reward = torch.ones_like(obs_buf[:, 0])
    return reward
```
- Reward constant de 1.0 (pas de reward spécifique dans la classe de base)

## 3. TERMINAISON

### Conditions de terminaison (compute_duckling_reset):

1. **Chute par contact**
```python
masked_contact_buf = contact_buf.clone()
masked_contact_buf[:, contact_body_ids, :] = 0
fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
```
- Contact détecté sur des parties non autorisées (tout sauf pieds)

2. **Chute par hauteur**
```python
body_height = rigid_body_pos[..., 2]
fall_height = body_height < termination_heights
fall_height[:, contact_body_ids] = False
fall_height = torch.any(fall_height, dim=-1)
```
- Hauteur du corps en dessous du seuil
- terminationHeight: -0.05
- headTerminationHeight: 0.3

3. **Fin d'épisode**
```python
reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
```
- episodeLength: 500 steps

## 4. PARAMÈTRES IMPORTANTS

### Configuration physique:
- **controlFrequencyInv**: 2 (30 Hz de fréquence de contrôle)
- **dt**: controlFrequencyInv × sim_dt
- **period**: 0.6 secondes (période de marche)
- **numAMPObsSteps**: 5 (historique pour AMP)

### Normalisation:
- **linearVelocityScale**: 0.5
- **angularVelocityScale**: 0.25
- **useAverageVelocities**: True (utilise moyennes sur une période)

### Limites de commande:
- **linear_x**: [-0.3, 0.3] m/s
- **linear_y**: [-0.3, 0.3] m/s
- **yaw**: [-0.2, 0.2] rad/s

### Corps clés:
- **keyBodies**: ["left_foot", "right_foot"]
- **contactBodies**: ["left_foot", "right_foot"]
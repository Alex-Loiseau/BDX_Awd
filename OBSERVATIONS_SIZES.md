# Tailles d'observations pour chaque mode

Basé sur l'analyse du code original dans `old_awd/env/tasks/`

## Architecture

**Base duckling_obs** = 51 dimensions :
- projected_gravity : 3
- dof_pos : 16
- dof_vel : 16
- prev_actions : 16

Chaque mode ajoute ses propres **task_obs** (si `enable_task_obs: True`)

## Tailles par mode

| Mode | Base | Task obs | Total | Fichier IsaacLab |
|------|------|----------|-------|------------------|
| Duckling (base) | 51 | 0 | **51** | `duckling_base_env.py` |
| DucklingAMP | 51 | 0 | **51** | `duckling_amp_env.py` (à créer) |
| DucklingAMPTask | 51 | 0 | **51** | `duckling_amp_task_env.py` (à créer) |
| DucklingCommand | 51 | 3 | **54** | `duckling_command_env.py` ✅ |
| DucklingHeading | 51 | 5 | **56** | `duckling_heading_env.py` (à créer) |
| DucklingPerturb | 51 | 0 | **51** | `duckling_perturb_env.py` (à créer) |
| DucklingViewMotion | 51 | 0 | **51** | `duckling_view_motion_env.py` (à créer) |

## Task obs détails

### Command (3 dims)
```python
task_obs = commands_scaled  # [vx, vy, vyaw]
```

### Heading (5 dims)
```python
task_obs = [
    local_tar_dir[0],      # 1
    local_tar_dir[1],      # 2
    tar_speed,             # 3
    local_tar_face_dir[0], # 4
    local_tar_face_dir[1], # 5
]
```

### Autres modes
Pas de task_obs (ou task_obs_size = 0)

## Configuration dans go_bdx_cfg.py

`num_observations: 51` (base seulement)

Chaque mode override dans son `__init__()` :
```python
base_obs_size = 51
task_obs_size = <mode_specific>
cfg.num_observations = base_obs_size + task_obs_size
```

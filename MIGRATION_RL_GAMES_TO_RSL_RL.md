# Migration de rl-games vers RSL-RL pour Isaac Lab

## Vue d'ensemble

Ce document détaille la migration complète du système d'entraînement de **rl-games** vers **RSL-RL** (Robot Systems Lab - Reinforcement Learning), le framework officiel d'Isaac Lab.

### Objectif
Migrer l'intégralité du code d'entraînement pour utiliser RSL-RL au lieu de rl-games, tout en conservant les mêmes fonctionnalités et résultats.

---

## 1. Architecture actuelle (rl-games)

### 1.1 Composants principaux

#### Fichiers actuels à migrer/remplacer:
- `awd_isaaclab/scripts/run_isaaclab.py` - Script principal d'entraînement
- `old_awd/learning/common_agent.py` - Agent de base (hérite de `a2c_continuous.A2CAgent`)
- `old_awd/learning/awd_agent.py` - Agent AWD (style imitatif)
- `old_awd/learning/amp_agent.py` - Agent AMP (Adversarial Motion Priors)
- `old_awd/learning/hrl_agent.py` - Agent HRL (Hierarchical RL)
- `old_awd/learning/*_players.py` - Players pour l'inférence
- `old_awd/learning/*_models.py` - Modèles de réseau
- `old_awd/learning/*_network_builder.py` - Constructeurs de réseau

#### Configuration actuelle:
- Fichiers YAML séparés pour environnement et entraînement
- Exemple: `old_awd/data/cfg/go_bdx/duckling_command.yaml` (env)
- Exemple: `old_awd/data/cfg/go_bdx/train/awd_duckling.yaml` (training)

### 1.2 Flux d'exécution actuel

```
run_isaaclab.py
  ├─> AppLauncher (Isaac Sim)
  ├─> gymnasium.make() -> Crée env IsaacLab
  ├─> RLGPUEnv wrapper -> Convertit Gymnasium -> rl-games API
  ├─> rl_games.Runner
  │   ├─> Enregistre agents custom (awd, amp, hrl)
  │   ├─> Enregistre players custom
  │   ├─> Enregistre models custom
  │   └─> Enregistre network builders custom
  └─> runner.run() -> Entraînement
```

### 1.3 Éléments clés à préserver

#### Agents personnalisés:
1. **CommonAgent** (PPO de base)
   - Normalisation input/value
   - Bounds loss
   - Central value network (optionnel)

2. **AWDAgent** (AMP with Diversity - Style imitatif)
   - Discriminateur pour imitation
   - Encodeur pour diversité
   - Replay buffer AMP
   - Latent space pour styles

3. **AMPAgent** (Adversarial Motion Priors)
   - Discriminateur uniquement
   - Replay buffer AMP
   - Motion priors

4. **HRLAgent** (Hierarchical RL)
   - Latent skills
   - High-level policy
   - Low-level policy

#### Hyperparamètres importants:
- PPO: `horizon_length=32`, `minibatch_size=16384`, `mini_epochs=6`
- Learning rate: `2e-5` (constant)
- AMP: `disc_coef=5`, `disc_reward_scale=2`
- AWD: `enc_coef=5`, `disc_reward_w=0.5`, `enc_reward_w=0.5`

---

## 2. Architecture cible (RSL-RL)

### 2.1 Structure RSL-RL

RSL-RL est structuré autour de:
- `rsl_rl.runners.OnPolicyRunner` - Gère la boucle d'entraînement
- `rsl_rl.algorithms.PPO` - Algorithme PPO
- `rsl_rl.modules.ActorCritic` - Réseau acteur-critique
- `rsl_rl.env.VecEnv` - Interface environnement vectorisé

### 2.2 Intégration Isaac Lab

Isaac Lab fournit:
- `isaaclab.utils.wrappers.rsl_rl.RslRlVecEnvWrapper` - Wrapper pour DirectRLEnv
- `isaaclab.utils.wrappers.rsl_rl.RslRlOnPolicyRunnerCfg` - Configuration runner
- Exemples dans `IsaacLab/source/standalone/workflows/rsl_rl/`

---

## 3. Plan de migration détaillé

### ✅ Phase 0: Préparation (COMPLÉTÉ)
- [x] Analyse du code actuel rl-games
- [x] Identification des composants à migrer
- [x] Création de ce document de suivi

### 🔄 Phase 1: Configuration de base

#### 1.1 Installer RSL-RL
- [ ] Vérifier si RSL-RL est déjà installé avec Isaac Lab
- [ ] Si non: `pip install rsl-rl` ou utiliser la version bundled

#### 1.2 Créer structure de configuration RSL-RL
- [ ] Créer `awd_isaaclab/configs/train/` pour configs d'entraînement
- [ ] Convertir configs YAML rl-games en dataclasses Python RSL-RL
- [ ] Créer `DucklingCommandPPORunnerCfg` (basé sur RslRlOnPolicyRunnerCfg)
- [ ] Créer `DucklingHeadingPPORunnerCfg`
- [ ] Créer configs pour autres tâches (Perturb, AMP, etc.)

**Fichiers à créer:**
- `awd_isaaclab/configs/train/duckling_command_ppo_cfg.py`
- `awd_isaaclab/configs/train/duckling_heading_ppo_cfg.py`
- `awd_isaaclab/configs/train/duckling_amp_cfg.py` (pour AMP)
- `awd_isaaclab/configs/train/duckling_awd_cfg.py` (pour AWD)

### 🔄 Phase 2: Script d'entraînement de base (PPO simple)

#### 2.1 Créer nouveau script train
- [ ] Créer `awd_isaaclab/scripts/train_rsl_rl.py`
- [ ] Implémenter AppLauncher pour Isaac Sim
- [ ] Créer environnement avec DirectRLEnv
- [ ] Wrapper avec `RslRlVecEnvWrapper`
- [ ] Créer `OnPolicyRunner` de RSL-RL
- [ ] Implémenter boucle d'entraînement

**Code de référence:**
```python
# Exemple structure
from isaaclab.app import AppLauncher
from isaaclab.utils.wrappers.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner

# 1. Launch Isaac Sim
launcher = AppLauncher(...)
simulation_app = launcher.app

# 2. Create environment
env = gymnasium.make(...)

# 3. Wrap for RSL-RL
env = RslRlVecEnvWrapper(env)

# 4. Create runner
runner_cfg = RslRlOnPolicyRunnerCfg(...)
runner = OnPolicyRunner(env, runner_cfg)

# 5. Train
runner.learn(num_learning_iterations=10000)
```

#### 2.2 Tester avec DucklingCommand
- [ ] Lancer entraînement simple PPO
- [ ] Vérifier convergence
- [ ] Comparer avec résultats rl-games

### 🔄 Phase 3: Migration agents custom (AWD, AMP, HRL)

#### 3.1 Analyser différences PPO
- [ ] Comparer `rl_games.A2CAgent` vs `rsl_rl.PPO`
- [ ] Identifier fonctionnalités manquantes dans RSL-RL
- [ ] Documenter adaptations nécessaires

#### 3.2 Créer CustomPPO pour AWD
- [ ] Créer `awd_isaaclab/learning/awd_ppo.py`
- [ ] Hériter de `rsl_rl.algorithms.PPO`
- [ ] Ajouter discriminateur (comme dans `awd_agent.py`)
- [ ] Ajouter encodeur pour diversité
- [ ] Ajouter replay buffer AMP
- [ ] Implémenter compute_disc_reward()
- [ ] Implémenter compute_enc_reward()
- [ ] Modifier loss pour inclure disc_loss + enc_loss

**Composants à porter:**
```python
# De old_awd/learning/awd_agent.py
- _amp_debug()
- _disc_loss()
- _enc_loss()
- _fetch_amp_obs_demo()
- _update_amp_demos()
- compute_disc_reward()
- compute_enc_reward()
```

#### 3.3 Créer CustomPPO pour AMP
- [ ] Créer `awd_isaaclab/learning/amp_ppo.py`
- [ ] Hériter de `rsl_rl.algorithms.PPO`
- [ ] Ajouter discriminateur uniquement
- [ ] Ajouter replay buffer AMP
- [ ] Implémenter compute_disc_reward()

#### 3.4 Créer CustomPPO pour HRL
- [ ] Créer `awd_isaaclab/learning/hrl_ppo.py`
- [ ] Implémenter low-level policy
- [ ] Implémenter high-level policy
- [ ] Gérer latent skills

### 🔄 Phase 4: Réseaux de neurones

#### 4.1 Analyser réseaux actuels
- [ ] Étudier `awd_network_builder.py`
- [ ] Étudier `amp_network_builder.py`
- [ ] Étudier `hrl_network_builder.py`

#### 4.2 Créer modules réseau RSL-RL
- [ ] Créer `awd_isaaclab/learning/networks/awd_actor_critic.py`
- [ ] Créer discriminateur réseau
- [ ] Créer encodeur réseau
- [ ] Hériter de `rsl_rl.modules.ActorCritic`

**Architecture à reproduire:**
```
Actor:
  - MLP: [1024, 1024, 512] + ReLU
  - Output: actions

Critic:
  - MLP: [1024, 1024, 512] + ReLU
  - Output: value

Discriminateur (AMP/AWD):
  - MLP: [1024, 1024, 512] + ReLU
  - Output: real/fake logit

Encodeur (AWD):
  - MLP: [1024, 512] + ReLU
  - Output: latent encoding
```

### 🔄 Phase 5: Configuration et hyperparamètres

#### 5.1 Mapper configs rl-games -> RSL-RL
- [ ] Créer tableau de correspondance des paramètres
- [ ] Adapter configs PPO

**Mapping initial:**
| rl-games | RSL-RL | Notes |
|----------|--------|-------|
| `horizon_length` | `num_steps_per_env` | Rollout length |
| `minibatch_size` | `num_mini_batches` | Calculé différemment |
| `mini_epochs` | `num_learning_epochs` | Epochs par update |
| `learning_rate` | `learning_rate` | Identique |
| `gamma` | `gamma` | Discount factor |
| `tau` | `lam` | GAE lambda |
| `e_clip` | `clip_param` | PPO clip |
| `entropy_coef` | `entropy_coef` | Identique |

#### 5.2 Créer dataclasses configuration
- [ ] `AWDPPOCfg` avec tous les hyperparams AWD
- [ ] `AMPPPOCfg` avec tous les hyperparams AMP
- [ ] `HRLPPOCfg` avec tous les hyperparams HRL

### 🔄 Phase 6: Players (Inférence)

#### 6.1 Créer players RSL-RL
- [ ] Analyser `old_awd/learning/*_players.py`
- [ ] Créer script d'inférence `awd_isaaclab/scripts/play_rsl_rl.py`
- [ ] Charger checkpoint RSL-RL
- [ ] Exécuter politique en mode eval

### 🔄 Phase 7: Utilitaires et logging

#### 7.1 TensorBoard logging
- [ ] Adapter logging pour RSL-RL
- [ ] Logger métriques custom (disc_loss, enc_loss, etc.)
- [ ] Logger rewards AMP/AWD

#### 7.2 Checkpointing
- [ ] Configurer sauvegarde checkpoints
- [ ] Implémenter best model saving
- [ ] Tester chargement checkpoints

### 🔄 Phase 8: Tests et validation

#### 8.1 Tests unitaires
- [ ] Tester chaque agent séparément
- [ ] Tester réseaux de neurones
- [ ] Tester compute_reward custom

#### 8.2 Tests d'entraînement
- [ ] DucklingCommand avec PPO simple
- [ ] DucklingCommand avec AWD
- [ ] DucklingHeading avec PPO
- [ ] DucklingAMP avec AMP

#### 8.3 Validation résultats
- [ ] Comparer courbes d'apprentissage rl-games vs RSL-RL
- [ ] Vérifier convergence
- [ ] Valider performance finale

### 🔄 Phase 9: Documentation

#### 9.1 Mise à jour docs
- [ ] Mettre à jour README avec instructions RSL-RL
- [ ] Documenter nouveaux scripts train/play
- [ ] Créer guide de migration pour utilisateurs

#### 9.2 Cleanup
- [ ] Supprimer code rl-games obsolète (optionnel)
- [ ] Nettoyer imports
- [ ] Vérifier dépendances requirements.txt

---

## 4. Correspondance des fichiers

### Old (rl-games) → New (RSL-RL)

| Ancien fichier | Nouveau fichier | Status |
|----------------|-----------------|--------|
| `awd_isaaclab/scripts/run_isaaclab.py` | `awd_isaaclab/scripts/train_rsl_rl.py` | ⏳ À créer |
| `old_awd/learning/common_agent.py` | `rsl_rl.algorithms.PPO` (base) | ✅ Built-in |
| `old_awd/learning/awd_agent.py` | `awd_isaaclab/learning/awd_ppo.py` | ⏳ À créer |
| `old_awd/learning/amp_agent.py` | `awd_isaaclab/learning/amp_ppo.py` | ⏳ À créer |
| `old_awd/learning/hrl_agent.py` | `awd_isaaclab/learning/hrl_ppo.py` | ⏳ À créer |
| `old_awd/learning/*_network_builder.py` | `awd_isaaclab/learning/networks/*.py` | ⏳ À créer |
| `old_awd/learning/*_players.py` | `awd_isaaclab/scripts/play_rsl_rl.py` | ⏳ À créer |
| `old_awd/data/cfg/*/train/*.yaml` | `awd_isaaclab/configs/train/*_cfg.py` | ⏳ À créer |

---

## 5. Détails techniques importants

### 5.1 Différences API clés

#### Environnement:
```python
# rl-games (ancien)
class RLGPUEnv(vecenv.IVecEnv):
    def step(self, action):
        # Retourne 4 valeurs: obs, reward, done, info
        return obs, reward, done, info

# RSL-RL (nouveau)
class RslRlVecEnvWrapper:
    def step(self, actions):
        # Retourne VecEnvStepReturn avec obs, privileged_obs, rewards, dones, infos
        return VecEnvStepReturn(...)
```

#### Observations:
```python
# rl-games: obs simple ou dict {"obs": obs, "states": states}
# RSL-RL: dict {"policy": obs} avec support privileged_obs
```

### 5.2 Gestion du replay buffer AMP

Dans rl-games (ancien):
```python
# old_awd/learning/awd_agent.py
self._amp_obs_demo_buffer  # Buffer des demos
self._amp_replay_buffer     # Replay buffer
```

Dans RSL-RL (nouveau):
- Créer classe `AMPReplayBuffer` custom
- Stocker dans agent custom AWD/AMP
- Utiliser lors du calcul disc_loss

### 5.3 Calcul des rewards

#### AWD (style imitatif):
```python
# Reward total = task_reward_w * task_rew + disc_reward_w * disc_rew + enc_reward_w * enc_rew
total_reward = (
    self.task_reward_w * task_rewards +
    self.disc_reward_w * disc_rewards +
    self.enc_reward_w * enc_rewards
)
```

#### AMP (imitation pure):
```python
# Reward total = disc_reward_scale * disc_rew
total_reward = self.disc_reward_scale * disc_rewards
```

---

## 6. Checklist de validation

### Avant de considérer la migration terminée:

- [ ] Tous les agents fonctionnent (PPO, AWD, AMP, HRL)
- [ ] Convergence comparable à rl-games
- [ ] Performance finale >= rl-games
- [ ] Checkpointing fonctionne
- [ ] Inférence (play) fonctionne
- [ ] Logs TensorBoard corrects
- [ ] Documentation à jour
- [ ] Tests passent
- [ ] Code nettoyé et commenté

---

## 7. Commandes de test

### Entraînement:
```bash
# PPO simple
./run_with_isaaclab.sh awd_isaaclab/scripts/train_rsl_rl.py \
    --task DucklingCommand --robot go_bdx --num_envs 4096

# AWD
./run_with_isaaclab.sh awd_isaaclab/scripts/train_rsl_rl.py \
    --task DucklingCommand --robot go_bdx --algo awd --num_envs 4096

# AMP
./run_with_isaaclab.sh awd_isaaclab/scripts/train_rsl_rl.py \
    --task DucklingAMP --robot go_bdx --algo amp --num_envs 4096
```

### Inférence:
```bash
./run_with_isaaclab.sh awd_isaaclab/scripts/play_rsl_rl.py \
    --task DucklingCommand --robot go_bdx --checkpoint runs/DucklingCommand_go_bdx/model.pt
```

---

## 8. Ressources et références

### Documentation:
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Isaac Lab workflows: `IsaacLab/source/standalone/workflows/rsl_rl/`
- Isaac Lab wrappers: `isaaclab/utils/wrappers/rsl_rl/`

### Exemples Isaac Lab:
- `train.py` - Entraînement de base
- `play.py` - Inférence
- Configuration examples dans envs

### Code de référence (ancien):
- `old_awd/run.py` - Structure principale
- `old_awd/learning/` - Tous les agents custom

---

## 9. Notes de progression

### [DATE] - Phase X
- Tâches complétées
- Problèmes rencontrés
- Solutions appliquées

---

**Statut général: 🔄 EN COURS - Phases 0-2 partiellement complétées**

## Statut détaillé par phase

### Phase 0: Préparation ✅ COMPLÉTÉ
- [x] Analyse du code actuel rl-games
- [x] Identification des composants à migrer
- [x] Création de ce document de suivi

### Phase 1: Configuration de base ✅ COMPLÉTÉ
- [x] RSL-RL installé (version 3.1.3)
- [x] tensorboard installé
- [x] Structure de configuration créée: `awd_isaaclab/configs/agents/`
- [x] Configuration PPO créée: `rsl_rl_ppo_cfg.py`

### Phase 2: Script d'entraînement de base ✅ PARTIELLEMENT COMPLÉTÉ
- [x] Script `train_rsl_rl.py` créé
- [x] Hyperparamètres PPO configurés (identiques à rl-games)
- [x] API DirectRLEnv corrigée (_apply_action implémenté)
- [ ] **PROBLÈME**: Environnement bloque au démarrage de l'entraînement

**Prochaine étape: Débugger le blocage de l'environnement pendant l'entraînement**

## Notes de débogage (2025-11-22)

### Problèmes identifiés:
1. L'environnement se crée correctement
2. RSL-RL OnPolicyRunner démarre
3. Le processus se bloque après les premiers warnings Gymnasium
4. Warnings sur types Tensor vs numpy (normal avec RslRlVecEnvWrapper)

### Solutions possibles à tester:
1. Désactiver le passive_env_checker de Gymnasium
2. Vérifier la compatibilité headless mode
3. Tester avec viewer activé pour voir si c'est un problème headless
4. Augmenter le timeout
5. Vérifier les logs Isaac Sim détaillés (/tmp/isaaclab_*.log)

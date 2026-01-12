# Explication Ultra-Détaillée du Code (RL-PDNN)

Ce document plonge au cœur du code. Nous allons décortiquer chaque fonction importante, ligne par ligne, pour comprendre la mécanique interne du projet.

---

## 1. L'Environnement : `rl_pdnn/env.py`

C'est ici que vit la simulation. C'est le monde virtuel dans lequel l'IA s'entraîne.

### `__init__` (Le Constructeur)
```python
self.action_space = spaces.Discrete(num_devices)
```
*   **Détail** : Définit que l'IA peut choisir un nombre entier entre 0 et `num_devices - 1`. Si vous avez 5 appareils, l'IA peut dire "0", "1", "2", "3" ou "4".

### `_get_observation` (Les Yeux de l'IA)
```python
def _get_observation(self):
    # 1. Normalisation de la couche actuelle
    obs = [float(self.current_layer_idx) / self.num_layers]
    
    for d in self.devices:
        # 2. Normalisation du CPU (On divise par 2.0 car c'est le max attendu)
        obs.append(d.cpu_speed / 2.0)
        # 3. Normalisation de la Mémoire (On divise par 500 Mo)
        obs.append((d.memory_capacity - d.current_memory_usage) / 500.0)
        # ...
    return np.array(obs, dtype=np.float32)
```
*   **Pourquoi diviser ?** : Les réseaux de neurones (le cerveau de l'IA) fonctionnent mal avec des chiffres comme "500" ou "1000". Ils préfèrent "0.5" ou "0.9". En divisant par la valeur maximale possible, on ramène tout entre 0 et 1. C'est crucial pour que l'apprentissage soit rapide.

### `step` (Le Pas de Temps)
C'est la fonction la plus complexe. Elle simule ce qui se passe quand l'IA prend une décision.

```python
# A. Vérification des Contraintes
if not selected_device.can_host(current_layer):
    reward = -100 
    reward -= 50 
```
*   **Logique** : Avant même de calculer le temps, on vérifie si c'est *possible*. L'appareil a-t-il assez de RAM ? A-t-il le droit (privacy) ? Si non, on donne une **énorme punition** (-150 points). C'est comme donner une décharge électrique à l'IA pour lui dire "Ne fais plus jamais ça !".

```python
# B. Calcul de la Latence de Calcul
comp_latency = current_layer.computation_demand / selected_device.cpu_speed
```
*   **Physique** : Temps = Travail / Vitesse. Si la couche demande 10 opérations et que l'appareil fait 2 opérations/seconde, ça prend 5 secondes.

```python
# C. Calcul de la Latence Réseau (Le Transfert)
if self.previous_device_id != -1 and self.previous_device_id != selected_device_id:
    trans_latency = prev_layer_output / selected_device.bandwidth
```
*   **Condition** : `Target != Previous`. Si on change d'appareil (ex: de l'Appareil 1 vers l'Appareil 2), on doit envoyer les données par le réseau.
*   **Physique** : Temps = Taille des Données / Vitesse Wifi. C'est souvent là que l'IA perd le plus de points. Elle apprendra qu'il vaut mieux rester sur le même appareil si possible pour éviter ce coût.

---

## 2. L'Agent : `rl_pdnn/agent.py`

C'est le cerveau qui apprend. Il utilise un algorithme appelé DQN (Deep Q-Network).

### `DQN` (Le Réseau de Neurones)
```python
self.fc1 = nn.Linear(input_dim, 256)
self.fc2 = nn.Linear(256, 256)
self.fc3 = nn.Linear(256, output_dim)
```
*   **Structure** : C'est un cerveau à 3 étages.
    *   **Entrée** : L'état du monde (CPU, Mémoire, Couche actuelle...).
    *   **Milieu (Caché)** : 256 neurones qui "réfléchissent". On a augmenté ce nombre (c'était 128) pour le rendre plus intelligent.
    *   **Sortie** : Une note pour chaque appareil possible. Si la sortie est `[10, 2, -5]`, l'IA choisira l'appareil 0 (car 10 est la meilleure note).

### `act` (La Prise de Décision - Exploration vs Exploitation)
```python
if np.random.rand() <= self.epsilon:
    return random.randrange(self.action_dim)
else:
    # Utiliser le réseau pour décider
```
*   **Le Dilemme** :
    *   `epsilon` commence à 1.0 (100% de hasard). L'IA est curieuse, elle essaie tout.
    *   `epsilon` diminue petit à petit.
    *   À la fin, l'IA devient sérieuse et utilise son expérience. Sans cette phase de hasard au début, elle pourrait passer à côté d'une stratégie géniale sans le savoir.

### `replay` (L'Apprentissage / La Révision)
C'est ici que la magie opère mathématiquement.

```python
# 1. On prend un lot de souvenirs (Batch)
minibatch = random.sample(self.memory, self.batch_size)

# 2. Q(s, a) : Ce que je pensais qu'il allait se passer
curr_q = self.policy_net(states).gather(1, actions)

# 3. Target : Ce qui s'est VRAIMENT passé + le futur espéré
# Récompense immédiate + (Gamma * Meilleur futur possible)
target_q = rewards + (self.gamma * next_q * (1 - dones))

# 4. Correction (Backpropagation)
loss = self.criterion(curr_q, target_q.detach())
loss.backward()
```
*   **Equation de Bellman** : C'est la formule mathématique au cœur du RL. Elle dit : *"La valeur d'une action, c'est la récompense immédiate PLUS la valeur de la meilleure action suivante."*
*   **Apprentissage** : L'agent compare sa prédiction (`curr_q`) avec la réalité (`target_q`). S'il s'est trompé, il modifie ses neurones (`loss.backward`) pour ne plus faire l'erreur.

---

## 3. Le Moteur d'Inférence : `integrated_system/inference_engine.py`

C'est le pont entre la théorie (RL) et la pratique (PyTorch).

```python
def run(self, input_data, allocation_map):
    x = input_data
    current_device = -1
    
    for i, layer in enumerate(self.model.layers):
        target_device = allocation_map[i]
        
        # Simulation du Réseau
        if target_device != current_device:
            # On calcule la taille réelle du tenseur PyTorch en Méga-Octets
            data_size_mb = x.element_size() * x.nelement() / (1024 * 1024)
            print(f"Transferring {data_size_mb} MB...")
            
        # Simulation du Calcul
        x = layer(x) # Exécution réelle de la couche PyTorch
```
*   **Analyse** : Ce code prend votre vrai modèle LeNet-5 et le découpe.
*   **`allocation_map[i]`** : C'est l'ordre donné par l'IA (ex: "Appareil 3").
*   **`layer(x)`** : C'est la vraie mathématique convolutionnelle. On ne fait pas semblant. Le tenseur `x` est transformé.
*   **Pourquoi c'est fort ?** : Parce qu'on mélange une *simulation* de temps (print "Transferring") avec une *exécution* réelle de code (PyTorch). Cela permet de valider que la stratégie de l'IA fonctionne sur un vrai programme.

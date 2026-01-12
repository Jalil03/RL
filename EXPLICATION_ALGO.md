# Comprendre notre Algorithme : RL-PDNN

Voici une explication claire et détaillée de ce que nous avons construit. Le but est de comprendre **pourquoi** ça marche et **comment** l'Intelligence Artificielle prend ses décisions.

---

## 1. Le Problème : L'IA sur les petits objets (IoT)
Imaginez que vous voulez faire tourner une grosse Intelligence Artificielle (comme la reconnaissance faciale) sur des petits objets connectés (caméras de surveillance, montres connectées, capteurs).
*   **Problème** : Ces objets sont "faibles" (peu de mémoire, processeur lent). Ils ne peuvent pas faire tourner tout le modèle eux-mêmes.
*   **Solution classique** : Tout envoyer dans le Cloud. Mais c'est lent et ça pose des problèmes de vie privée (privacy).

## 2. Notre Solution : "Diviser pour Régner"
L'idée de **RL-PDNN** est de **découper** le modèle d'IA (le CNN) en plusieurs tranches (les "couches").
*   Au lieu de tout faire sur un seul appareil, on distribue le travail :
    *   *Appareil 1* fait un peu de calcul.
    *   Il passe le relais à *Appareil 2*.
    *   Etc.

Mais **qui décide** quel appareil fait quoi ? C'est le rôle de notre algorithme.

---

## 3. Le Cerveau : L'Apprentissage par Renforcement (RL)

Nous utilisons un "Agent" (un robot virtuel) qui apprend par l'expérience, exactement comme on dresse un chien avec des récompenses.

### A. Ce qu'il voit (L'État / State)
À chaque étape (pour chaque couche du réseau de neurones), l'Agent regarde la situation globale :
1.  **"Où en suis-je ?"** : Je dois placer la couche n°3.
2.  **"Qui est disponible ?"** :
    *   L'appareil A est rapide mais a peu de mémoire.
    *   L'appareil B est lent mais très sécurisé.
    *   L'appareil C est surchargé.

### B. Ce qu'il fait (L'Action / Action)
L'Agent prend une décision simple :
> *"Je décide que la couche n°3 sera exécutée par l'appareil B."*

### C. Sa Note (La Récompense / Reward)
Une fois la décision prise, on calcule le résultat :
*   **Si ça s'est bien passé** : On chronomètre le temps (calcul + transfert Wifi).
    *   *Note = -Temps (Moins on perd de temps, meilleure est la note).*
*   **Si ça s'est mal passé** (ex: l'appareil B n'avait plus de mémoire ou n'avait pas le droit de voir ces données) :
    *   *Note = -50 (Grosse punition !)*

---

## 4. L'Entraînement (Training)

Au début (Épisode 0), l'Agent est "bête". Il place les couches au hasard. Il fait n'importe quoi et reçoit de très mauvaises notes (-500).

Mais il a une mémoire (**Replay Buffer**). Il se souvient : *"Tiens, quand j'ai mis la grosse couche sur le petit appareil, j'ai été puni. Je ne le referai plus."*

Au bout de **500 épisodes** (comme dans notre simulation), il a tout compris. Il connait par cœur les forces et faiblesses de chaque appareil. Il est capable de tracer le **chemin parfait** pour que le calcul soit le plus rapide possible sans jamais planter.

## 5. Résumé de votre Projet

1.  **`rl_pdnn/main.py`** : C'est l'école. L'agent s'entraîne virtuellement pendant 500 essais.
2.  **`split_inference/train_cnn.py`** : On fabrique le "vrai" travail (le modèle LeNet-5 qui reconnaît les chiffres).
3.  **`full_demo.py`** : L'examen final.
    *   L'Agent (maintenant intelligent) regarde le LeNet-5.
    *   Il dit : *"Toi couche 1 va là-bas, toi couche 2 viens ici..."*
    *   Le système exécute les ordres.
    *   L'image est reconnue (Classe 5 !), et on a gagné du temps.

C'est ça, la puissance du **Reinforcement Learning** appliqué aux systèmes distribués.

---

## 6. Explication Détaillée du Code

Voici une promenade guidée à travers les fichiers les plus importants pour comprendre la magie du code.

### A. L'Environnement (`rl_pdnn/env.py`)
Ce fichier simule le monde réel. C'est ici que sont définies les "règles du jeu".

*   **`_get_observation(self)`** [Ligne 49]
    *   C'est les yeux de l'IA. Elle construit un vecteur (une liste de chiffres) qui décrit la situation.
    *   On **normalise** les valeurs (diviser par 500 ou 100) pour que tous les chiffres soient entre 0 et 1. Les réseaux de neurones apprennent beaucoup mieux ainsi.
*   **`step(self, action)`** [Ligne 66]
    *   C'est le cœur de la simulation.
    *   **Pénalité (-50)** : Si l'IA choisit un appareil qui n'a pas assez de mémoire (`can_host` retourne Faux), on la punit sévèrement [Ligne 92].
    *   **Calcul de Latence** : On additionne le temps de calcul (`comp_latency`) et le temps de transfert wifi (`trans_latency`) si on change d'appareil [Ligne 121].
    *   **Récompense** : La note finale est `-total_latency`. Plus le temps est court (petit chiffre), plus la note est proche de 0 (ce qui est bien, car c'est négatif).

### B. L'Intelligent (Agent) (`rl_pdnn/agent.py`)
Ce fichier contient le cerveau.

*   **`class DQN(nn.Module)`** [Ligne 8]
    *   C'est le réseau de neurones artificiel.
    *   On utilise 3 couches (Input -> 256 neurones -> 256 neurones -> Output). Plus il y a de neurones, plus il peut comprendre de stratégies complexes.
*   **`act(self, state)`** [Ligne 54]
    *   C'est la prise de décision.
    *   Parfois il explore (choisit au hasard) grâce à `epsilon`.
    *   Sinon, il utilise son réseau (`self.policy_net`) pour prédire la meilleure action.
*   **`replay(self)`** [Ligne 66]
    *   C'est le moment où il révise ses leçons. Il prend un lot de souvenirs (`minibatch`) et corrige ses erreurs en ajustant les poids du réseau de neurones (`loss.backward()`).

### C. Le Moteur d'Exécution (`integrated_system/inference_engine.py`)
C'est le pont entre la simulation et la réalité.

*   **`run(self, input_data, allocation_map)`** [Ligne 14]
    *   Il prend le vrai modèle (LeNet-5) et la carte au trésor (la liste des décisions de l'IA).
    *   Il boucle sur chaque couche (`for i, layer in enumerate...`).
    *   Si l'IA a dit que la couche 1 est sur l'appareil 0 et la couche 2 sur l'appareil 4, le moteur détecte le changement :
        ```python
        if target_device != current_device:
             print("Transferring...")
        ```
    *   Il simule alors le transfert de données avant d'exécuter la couche suivante.


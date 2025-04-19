"""
BOME (Balancing and Optimisation Matrix Engine)
Formalisation algorithmique exportable
"""

class BOMEEngine:
    def __init__(self, config=None):
        """
        Initialisation du moteur BOME

        Args:
            config: Configuration initiale des paramètres BOME
        """
        # Paramètres par défaut
        self.config = {
            # Paramètres généraux
            'neutral_value': 0.5,               # Valeur d'équilibre neutre
            'min_value': 0.1,                   # Valeur minimale autorisée
            'max_value': 0.9,                   # Valeur maximale autorisée

            # Classification contextuelle
            'context_thresholds': {
                'stable_low': 0.05,             # Seuil pour contexte stable/basse volatilité
                'stable_high': 0.06,            # Seuil pour contexte stable/haute volatilité
                'volatile_low': 0.07,           # Seuil pour contexte volatile/basse intensité
                'volatile_high': 0.09,          # Seuil pour contexte volatile/haute intensité
            },

            # Réponse proportionnelle
            'base_amplitude': 1.0,              # Amplitude de base
            'max_amplitude_factor': 2.0,        # Facteur maximal d'amplitude
            'signal_amplification': {
                'stable_low': 1.0,
                'stable_high': 1.2,
                'volatile_low': 0.8,
                'volatile_high': 0.6,
            },

            # Intégration des coûts
            'cost_multiplier': 2.0,             # Multiplicateur de pénalité des coûts

            # Protocole post-intervention
            'observation_periods': {
                'stable_low': 7,                # Jours/périodes d'observation
                'stable_high': 5,
                'volatile_low': 3,
                'volatile_high': 2,
            },
            'recovery_periods': {
                'stable_low': 14,               # Jours/périodes de récupération
                'stable_high': 10,
                'volatile_low': 7,
                'volatile_high': 5,
            },

            # Calibration adaptative
            'learning_rate': 0.05,              # Taux d'apprentissage pour l'adaptation
            'explore_rate': 0.10,               # Taux d'exploration pour l'optimisation bayésienne
        }

        # Écrasement des paramètres par défaut si fournis
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value

        # État interne
        self.last_intervention_time = None
        self.last_intervention_value = self.config['neutral_value']
        self.current_observation_period = 0
        self.current_recovery_period = 0
        self.current_phase = 'stable_low'
        self.intervention_history = []

    def analyze_context(self, data, current_time):
        """
        Analyse le contexte actuel pour déterminer la phase

        Args:
            data: Données historiques récentes
            current_time: Temps/index actuel

        Returns:
            Phase de contexte identifiée
        """
        # À implémenter selon domaine spécifique
        # Exemple simplifié:
        volatility = self._calculate_volatility(data)
        trend = self._calculate_trend(data)

        if trend > 0.1:  # Tendance positive
            if volatility > 0.15:
                return 'volatile_high'
            else:
                return 'stable_high'
        elif trend < -0.1:  # Tendance négative
            if volatility > 0.15:
                return 'volatile_low'
            else:
                return 'stable_low'
        else:  # Consolidation
            if volatility > 0.15:
                return 'volatile_high'
            else:
                return 'stable_low'

    def should_intervene(self, signal, context_phase):
        """
        Détermine si une intervention est nécessaire

        Args:
            signal: Force du signal (écart normalisé)
            context_phase: Phase de contexte actuelle

        Returns:
            Boolean indiquant si intervention nécessaire
        """
        threshold = self.config['context_thresholds'][context_phase]
        return abs(signal) > threshold

    def calculate_intervention_amplitude(self, signal, context_phase):
        """
        Calcule l'amplitude optimale de l'intervention

        Args:
            signal: Force du signal (écart normalisé)
            context_phase: Phase de contexte actuelle

        Returns:
            Amplitude d'intervention (0.0 à 1.0)
        """
        threshold = self.config['context_thresholds'][context_phase]
        base_amplitude = self.config['base_amplitude']
        amplification = self.config['signal_amplification'][context_phase]
        max_factor = self.config['max_amplitude_factor']

        intensity_factor = min(max_factor, abs(signal) / threshold)
        amplitude = base_amplitude * amplification * intensity_factor

        return amplitude

    def evaluate_cost_benefit(self, potential_benefit, intervention_cost):
        """
        Évalue le ratio coût-bénéfice de l'intervention

        Args:
            potential_benefit: Bénéfice potentiel estimé
            intervention_cost: Coût de l'intervention

        Returns:
            Score combiné coût-bénéfice
        """
        cost_multiplier = self.config['cost_multiplier']
        return potential_benefit - (intervention_cost * cost_multiplier)

    def get_intervention_decision(self, current_value, target_value, data, current_time):
        """
        Point d'entrée principal: détermine si et comment intervenir

        Args:
            current_value: Valeur actuelle
            target_value: Valeur cible/recommandée
            data: Données historiques récentes
            current_time: Temps/index actuel

        Returns:
            Tuple (intervene, new_value, rationale)
        """
        # 1. Analyse du contexte
        context_phase = self.analyze_context(data, current_time)
        self.current_phase = context_phase

        # 2. Calcul du signal (écart normalisé)
        neutral = self.config['neutral_value']
        signal = (target_value - current_value) / (self.config['max_value'] - self.config['min_value'])

        # 3. Vérification de la période d'observation
        in_observation_period = False
        if self.last_intervention_time is not None:
            time_since_intervention = current_time - self.last_intervention_time
            observation_period = self.config['observation_periods'][context_phase]

            if time_since_intervention < observation_period:
                in_observation_period = True

        # 4. Décision d'intervention
        if in_observation_period:
            # En période d'observation, maintien de la valeur actuelle
            return (False, current_value, "En période d'observation")

        # Vérification si intervention nécessaire
        if not self.should_intervene(signal, context_phase):
            # Signal trop faible pour intervention
            if self.last_intervention_time is not None:
                # Vérifier si en période de récupération
                time_since_intervention = current_time - self.last_intervention_time
                observation_period = self.config['observation_periods'][context_phase]
                recovery_period = self.config['recovery_periods'][context_phase]

                if time_since_intervention >= observation_period:
                    # En période de récupération, retour progressif vers valeur neutre
                    recovery_progress = min(1.0, (time_since_intervention - observation_period) / recovery_period)
                    new_value = self.last_intervention_value + (neutral - self.last_intervention_value) * recovery_progress
                    return (True, new_value, "Retour progressif vers équilibre")

            return (False, current_value, "Signal insuffisant")

        # 5. Calcul de l'amplitude d'intervention
        amplitude = self.calculate_intervention_amplitude(signal, context_phase)

        # Direction de l'intervention
        if signal > 0:
            # Augmentation
            adjustment = (self.config['max_value'] - neutral) * amplitude
            new_value = neutral + adjustment
        else:
            # Diminution
            adjustment = (neutral - self.config['min_value']) * amplitude
            new_value = neutral - adjustment

        # Limites des valeurs
        new_value = max(self.config['min_value'], min(self.config['max_value'], new_value))

        # 6. Évaluation coût-bénéfice
        benefit = abs(target_value - current_value)
        cost = abs(new_value - current_value) * 0.01  # Exemple simplifié de coût

        score = self.evaluate_cost_benefit(benefit, cost)

        if score <= 0:
            return (False, current_value, "Ratio coût-bénéfice défavorable")

        # 7. Enregistrement de l'intervention
        self.last_intervention_time = current_time
        self.last_intervention_value = new_value
        self.intervention_history.append({
            'time': current_time,
            'old_value': current_value,
            'new_value': new_value,
            'context': context_phase,
            'signal': signal,
            'amplitude': amplitude,
            'score': score
        })

        return (True, new_value, f"Intervention avec amplitude {amplitude:.2f}")

    def update_parameters(self, performance_metrics):
        """
        Met à jour les paramètres BOME selon les performances observées

        Args:
            performance_metrics: Métriques de performance récentes
        """
        # Implémentation simplifiée de l'optimisation bayésienne
        # À adapter selon le domaine spécifique
        pass

    def _calculate_volatility(self, data):
        """Calcule la volatilité des données récentes"""
        # Implémentation à adapter selon le domaine
        if len(data) < 2:
            return 0

        # Exemple simplifié: écart-type normalisé
        import numpy as np
        return np.std(data) / np.mean(data) if np.mean(data) != 0 else 0

    def _calculate_trend(self, data):
        """Calcule la tendance des données récentes"""
        # Implémentation à adapter selon le domaine
        if len(data) < 2:
            return 0

        # Exemple simplifié: rendement/changement relatif
        return (data[-1] - data[0]) / data[0] if data[0] != 0 else 0

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration spécifique au domaine
    config = {
        'neutral_value': 0.5,
        'min_value': 0.0,
        'max_value': 1.0,
        'context_thresholds': {
            'stable_low': 0.05,
            'stable_high': 0.07,
            'volatile_low': 0.06,
            'volatile_high': 0.09,
        }
    }

    # Initialisation du moteur BOME
    bome = BOMEEngine(config)

    # Données de test
    data = [0.51, 0.52, 0.53, 0.54, 0.57, 0.61, 0.62]
    current_value = 0.62
    target_value = 0.50
    current_time = 10

    # Obtention de la décision d'intervention
    intervene, new_value, rationale = bome.get_intervention_decision(
        current_value, target_value, data, current_time
    )

    print(f"Décision: {'Intervenir' if intervene else 'Ne pas intervenir'}")
    print(f"Nouvelle valeur: {new_value:.4f}")
    print(f"Justification: {rationale}")
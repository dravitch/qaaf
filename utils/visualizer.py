"""
Module de visualisation pour QAAF.
Fournit des fonctions pour générer des graphiques et visualisations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict,List,Tuple,Optional,Union,Any
from datetime import datetime

# Configuration du logging
logger=logging.getLogger (__name__)

# Configuration du style par défaut
sns.set_style ("whitegrid")
plt.rcParams['figure.figsize']=(15,8)
plt.rcParams['axes.grid']=True


def plot_performance_comparison (portfolio_values: pd.Series,
                                 asset_data: Dict[str,pd.DataFrame],
                                 allocations: Optional[pd.Series] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 benchmark_values: Optional[Dict[str,pd.Series]] = None,
                                 log_scale: bool = False,
                                 title: str = "Performance Comparée",
                                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualise la performance d'un portefeuille par rapport aux actifs sous-jacents.

    Args:
        portfolio_values: Série temporelle des valeurs du portefeuille
        asset_data: Dictionnaire des DataFrames pour chaque actif (avec colonne 'close')
        allocations: Série temporelle des allocations (optionnel)
        start_date: Date de début pour la visualisation (optionnel)
        end_date: Date de fin pour la visualisation (optionnel)
        benchmark_values: Dictionnaire des séries de valeurs pour les benchmarks (optionnel)
        log_scale: Utiliser une échelle logarithmique pour l'axe y
        title: Titre du graphique
        save_path: Chemin pour sauvegarder le graphique (optionnel)

    Returns:
        Figure matplotlib
    """
    # Création de la figure
    if allocations is not None:
        fig,(ax1,ax2)=plt.subplots (2,1,figsize=(15,10),gridspec_kw={'height_ratios':[3,1]})
    else:
        fig,ax1=plt.subplots (figsize=(15,8))

    # Filtrage des dates si spécifiées
    if start_date or end_date:
        mask=pd.Series (True,index=portfolio_values.index)
        if start_date:
            mask=mask & (portfolio_values.index >= pd.Timestamp (start_date))
        if end_date:
            mask=mask & (portfolio_values.index <= pd.Timestamp (end_date))

        portfolio_values=portfolio_values[mask]

        if allocations is not None:
            allocations=allocations[mask] if not allocations.empty else allocations

    # Normalisation pour la comparaison
    start_value=portfolio_values.iloc[0]
    portfolio_norm=portfolio_values / start_value

    # Tracé du portefeuille
    ax1.plot (portfolio_values.index,portfolio_norm,'b-',linewidth=2,label='Portefeuille')

    # Tracé des actifs sous-jacents
    for asset_name,df in asset_data.items ():
        # Filtrage et normalisation
        asset_close=df['close'].reindex (portfolio_values.index)

        # Si données manquantes
        if asset_close.iloc[0] == 0 or pd.isna (asset_close.iloc[0]):
            start_idx=(asset_close != 0).idxmax ()
        else:
            start_idx=asset_close.index[0]

        asset_norm=asset_close / asset_close.loc[start_idx]

        # Tracé de l'actif
        ax1.plot (asset_norm.index,asset_norm,'--',linewidth=1.5,label=asset_name,alpha=0.7)

    # Tracé des benchmarks si fournis
    if benchmark_values:
        for bench_name,bench_series in benchmark_values.items ():
            # Filtrage et normalisation
            bench_norm=bench_series.reindex (portfolio_values.index) / bench_series.iloc[0]

            # Tracé du benchmark
            ax1.plot (bench_norm.index,bench_norm,'-.',linewidth=1.5,
                      label=bench_name,alpha=0.6)

    # Configuration du graphique principal
    ax1.set_title (title)
    ax1.set_ylabel ('Performance (base 100)' if not log_scale else 'Performance (log)')
    ax1.legend (loc='best')
    ax1.grid (True)

    if log_scale:
        ax1.set_yscale ('log')

    # Tracé des allocations si fournies
    if allocations is not None and not allocations.empty:
        ax2.plot (allocations.index,allocations,'g-',linewidth=1.5)
        ax2.set_title ('Allocation')
        ax2.set_xlabel ('Date')
        ax2.set_ylabel ('Allocation')
        ax2.set_ylim (0,1)
        ax2.grid (True)

    plt.tight_layout ()

    # Sauvegarde si chemin spécifié
    if save_path:
        try:
            plt.savefig (save_path,dpi=300,bbox_inches='tight')
            logger.info (f"Graphique sauvegardé à {save_path}")
        except Exception as e:
            logger.warning (f"Échec de la sauvegarde du graphique: {str (e)}")

    return fig


def plot_metrics (metrics: Dict[str,pd.Series],
                  market_phases: Optional[pd.Series] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  highlight_dates: Optional[List[pd.Timestamp]] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualise l'évolution des métriques QAAF dans le temps.

    Args:
        metrics: Dictionnaire des séries temporelles pour chaque métrique
        market_phases: Série temporelle des phases de marché (optionnel)
        start_date: Date de début pour la visualisation (optionnel)
        end_date: Date de fin pour la visualisation (optionnel)
        highlight_dates: Liste des dates à mettre en évidence (optionnel)
        save_path: Chemin pour sauvegarder le graphique (optionnel)

    Returns:
        Figure matplotlib
    """
    # Nombre de métriques
    n_metrics=len (metrics)

    # Création de la figure
    fig,axes=plt.subplots (n_metrics,1,figsize=(15,4 * n_metrics),sharex=True)

    # Si une seule métrique, axes n'est pas un tableau
    if n_metrics == 1:
        axes=[axes]

    # Filtrage des dates si spécifiées
    if start_date or end_date:
        for metric_name,metric_series in metrics.items ():
            mask=pd.Series (True,index=metric_series.index)
            if start_date:
                mask=mask & (metric_series.index >= pd.Timestamp (start_date))
            if end_date:
                mask=mask & (metric_series.index <= pd.Timestamp (end_date))

            metrics[metric_name]=metric_series[mask]

        if market_phases is not None:
            mask=pd.Series (True,index=market_phases.index)
            if start_date:
                mask=mask & (market_phases.index >= pd.Timestamp (start_date))
            if end_date:
                mask=mask & (market_phases.index <= pd.Timestamp (end_date))

            market_phases=market_phases[mask]

    # Pour chaque métrique
    for i,(metric_name,metric_series) in enumerate (metrics.items ()):
        ax=axes[i]

        # Tracé de la métrique
        ax.plot (metric_series.index,metric_series,linewidth=1.5)

        # Coloration du fond selon les phases de marché si fournies
        if market_phases is not None:
            phase_colors={
                'bullish_low_vol':'#d8f3dc',  # Vert clair
                'bullish_high_vol':'#95d5b2',  # Vert moyen
                'bearish_low_vol':'#ffccd5',  # Rouge clair
                'bearish_high_vol':'#ff8fa3',  # Rouge moyen
                'consolidation_low_vol':'#f1faee',  # Bleu très clair
                'consolidation_high_vol':'#a8dadc'  # Bleu clair
            }

            # Récupération des phases uniques et leurs indices
            phases=market_phases.to_frame ()
            phases['phase_change']=phases.ne (phases.shift ())

            # Points de changement de phase
            change_points=phases[phases['phase_change']].index.tolist ()
            change_points=[metric_series.index[0]] + change_points + [metric_series.index[-1]]

            # Coloration de chaque segment
            for j in range (len (change_points) - 1):
                start_idx=change_points[j]
                end_idx=change_points[j + 1]

                if start_idx in market_phases.index:
                    phase=market_phases.loc[start_idx]
                    color=phase_colors.get (phase,'#f8f9fa')  # Gris par défaut

                    # Ajout de la zone colorée
                    ax.axvspan (start_idx,end_idx,color=color,alpha=0.4)

        # Mise en évidence des dates spécifiques
        if highlight_dates:
            for date in highlight_dates:
                if date in metric_series.index:
                    ax.axvline (x=date,color='red',linestyle='--',alpha=0.7)

        # Configuration du graphique
        ax.set_title (f"{metric_name}")
        ax.set_ylabel ('Valeur')
        ax.grid (True)

        # Ajout des valeurs moyennes et écarts-types
        mean_val=metric_series.mean ()
        std_val=metric_series.std ()

        ax.axhline (y=mean_val,color='gray',linestyle='-',alpha=0.7,label=f'Moyenne: {mean_val:.3f}')
        ax.axhline (y=mean_val + std_val,color='gray',linestyle='--',alpha=0.5,
                    label=f'+1 écart-type: {mean_val + std_val:.3f}')
        ax.axhline (y=mean_val - std_val,color='gray',linestyle='--',alpha=0.5,
                    label=f'-1 écart-type: {mean_val - std_val:.3f}')

        ax.legend (loc='best')

    # Titre global
    plt.suptitle ("Évolution des Métriques QAAF",fontsize=16)
    plt.tight_layout ()

    # Sauvegarde si chemin spécifié
    if save_path:
        try:
            plt.savefig (save_path,dpi=300,bbox_inches='tight')
            logger.info (f"Graphique sauvegardé à {save_path}")
        except Exception as e:
            logger.warning (f"Échec de la sauvegarde du graphique: {str (e)}")

    return fig


def plot_correlation_matrix (metrics: Dict[str,pd.Series],
                             market_phases: Optional[pd.Series] = None,
                             split_by_phase: bool = True,
                             annot: bool = True,
                             cmap: str = 'coolwarm',
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualise la matrice de corrélation entre les métriques.

    Args:
        metrics: Dictionnaire des séries temporelles pour chaque métrique
        market_phases: Série temporelle des phases de marché (optionnel)
        split_by_phase: Si True, crée une matrice par phase de marché
        annot: Afficher les valeurs de corrélation
        cmap: Palette de couleurs
        save_path: Chemin pour sauvegarder le graphique (optionnel)

    Returns:
        Figure matplotlib
    """
    # Conversion en DataFrame
    metrics_df=pd.DataFrame ({name:series for name,series in metrics.items ()})

    # Si pas de division par phase ou pas de phases fournies
    if not split_by_phase or market_phases is None:
        # Calcul de la matrice de corrélation
        corr_matrix=metrics_df.corr ()

        # Création de la figure
        plt.figure (figsize=(10,8))

        # Tracé de la heatmap
        sns.heatmap (corr_matrix,annot=annot,cmap=cmap,vmin=-1,vmax=1,
                     linewidths=.5,fmt=".2f")

        plt.title ("Matrice de Corrélation des Métriques")
        plt.tight_layout ()

        # Sauvegarde si chemin spécifié
        if save_path:
            try:
                plt.savefig (save_path,dpi=300,bbox_inches='tight')
                logger.info (f"Graphique sauvegardé à {save_path}")
            except Exception as e:
                logger.warning (f"Échec de la sauvegarde du graphique: {str (e)}")

        return plt.gcf ()

    # Division par phase
    else:
        # Ajout des phases au DataFrame
        metrics_df['market_phase']=market_phases.reindex (metrics_df.index)

        # Récupération des phases uniques
        unique_phases=metrics_df['market_phase'].dropna ().unique ()
        n_phases=len (unique_phases)

        # Nombre de colonnes pour le subplot
        n_cols=min (3,n_phases)
        n_rows=(n_phases + n_cols - 1) // n_cols

        # Création de la figure
        fig=plt.figure (figsize=(15,5 * n_rows))

        # Pour chaque phase
        for i,phase in enumerate (unique_phases):
            # Filtrage par phase
            phase_df=metrics_df[metrics_df['market_phase'] == phase].drop ('market_phase',axis=1)

            # Calcul de la matrice de corrélation
            corr_matrix=phase_df.corr ()

            # Création du subplot
            ax=fig.add_subplot (n_rows,n_cols,i + 1)

            # Tracé de la heatmap
            sns.heatmap (corr_matrix,annot=annot,cmap=cmap,vmin=-1,vmax=1,
                         linewidths=.5,ax=ax,fmt=".2f")

            ax.set_title (f"Phase: {phase}")

        plt.suptitle ("Matrices de Corrélation par Phase de Marché",fontsize=16)
        plt.tight_layout ()

        # Sauvegarde si chemin spécifié
        if save_path:
            try:
                plt.savefig (save_path,dpi=300,bbox_inches='tight')
                logger.info (f"Graphique sauvegardé à {save_path}")
            except Exception as e:
                logger.warning (f"Échec de la sauvegarde du graphique: {str (e)}")

        return fig


def plot_optimization_results (optimization_results: List[Dict],
                               key_metrics: List[str] = ['total_return','max_drawdown','sharpe_ratio'],
                               n_best: int = 50,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualise les résultats d'optimisation.

    Args:
        optimization_results: Liste des résultats d'optimisation triés
        key_metrics: Liste des métriques clés à visualiser
        n_best: Nombre des meilleurs résultats à afficher
        save_path: Chemin pour sauvegarder le graphique (optionnel)

    Returns:
        Figure matplotlib
    """
    if not optimization_results:
        logger.warning ("Aucun résultat d'optimisation à visualiser")
        return None

    # Limitation aux n meilleurs résultats
    top_results=optimization_results[:min (n_best,len (optimization_results))]

    # Préparation des données
    data=[]
    for i,result in enumerate (top_results):
        result_data={'rank':i + 1,'score':result.get ('score',0)}

        # Ajout des métriques
        metrics=result.get ('metrics',{})
        for metric in key_metrics:
            if metric in metrics:
                result_data[metric]=metrics[metric]

        # Ajout des paramètres principaux
        params=result.get ('params',{})
        for param_name,param_value in params.items ():
            if param_name in ['min_btc_allocation','max_btc_allocation','sensitivity',
                              'rebalance_threshold','vol_ratio_weight','bound_coherence_weight',
                              'alpha_stability_weight','spectral_score_weight']:
                result_data[param_name]=param_value

        data.append (result_data)

    # Conversion en DataFrame
    results_df=pd.DataFrame (data)

    # Création de la figure
    fig=plt.figure (figsize=(15,12))

    # 1. Tracé des métriques clés
    for i,metric in enumerate (key_metrics):
        if metric in results_df.columns:
            ax=fig.add_subplot (len (key_metrics) + 1,1,i + 1)

            # Tracé des valeurs
            ax.plot (results_df['rank'],results_df[metric],'o-',markersize=3)

            # Configuration
            ax.set_title (f"{metric}")
            ax.set_xlabel ('Rang')
            ax.set_ylabel ('Valeur')
            ax.grid (True)

            # Mise en évidence du meilleur résultat
            best_value=results_df[metric].iloc[0]
            ax.plot (1,best_value,'ro',markersize=8,label=f'Meilleur: {best_value:.2f}')

            ax.legend ()

    # 2. Tracé des poids des métriques
    weight_columns=[col for col in results_df.columns if col.endswith ('_weight')]

    if weight_columns:
        ax=fig.add_subplot (len (key_metrics) + 1,1,len (key_metrics) + 1)

        for weight_col in weight_columns:
            ax.plot (results_df['rank'],results_df[weight_col],'o-',label=weight_col,markersize=3)

        ax.set_title ("Poids des métriques")
        ax.set_xlabel ('Rang')
        ax.set_ylabel ('Poids')
        ax.grid (True)
        ax.legend ()

    plt.suptitle ("Analyse des Résultats d'Optimisation",fontsize=16)
    plt.tight_layout ()

    # Sauvegarde si chemin spécifié
    if save_path:
        try:
            plt.savefig (save_path,dpi=300,bbox_inches='tight')
            logger.info (f"Graphique sauvegardé à {save_path}")
        except Exception as e:
            logger.warning (f"Échec de la sauvegarde du graphique: {str (e)}")

    return fig


def save_performance_summary (results: Dict[str,Any],
                              comparison: Optional[Dict[str,Dict[str,float]]] = None,
                              file_path: str = None) -> str:
    """
    Génère un résumé des performances au format texte ou HTML.

    Args:
        results: Dictionnaire des résultats de performance
        comparison: Dictionnaire des benchmarks pour comparaison
        file_path: Chemin pour sauvegarder le résumé (optionnel)

    Returns:
        Résumé formaté
    """
    # Création du résumé
    summary="# Résumé des Performances QAAF\n\n"
    summary+=f"Date: {datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')}\n\n"

    # Métriques principales
    summary+="## Métriques de Performance\n\n"
    summary+="| Métrique | Valeur |\n"
    summary+="|----------|-------|\n"

    metrics=results.get ('metrics',{})
    for metric_name,metric_value in metrics.items ():
        # Formatage selon le type de métrique
        if isinstance (metric_value,(int,float)):
            if metric_name in ['total_return','volatility','max_drawdown']:
                formatted_value=f"{metric_value:.2f}%"
            else:
                formatted_value=f"{metric_value:.2f}"
        else:
            formatted_value=str (metric_value)

        summary+=f"| {metric_name} | {formatted_value} |\n"

    # Comparaison avec les benchmarks
    if comparison:
        summary+="\n## Comparaison avec les Benchmarks\n\n"
        summary+="| Stratégie | Rendement | Drawdown | Sharpe |\n"
        summary+="|-----------|-----------|----------|--------|\n"

        # QAAF d'abord
        qaaf_metrics=metrics
        summary+=f"| **QAAF** | {qaaf_metrics.get ('total_return',0):.2f}% | {qaaf_metrics.get ('max_drawdown',0):.2f}% | {qaaf_metrics.get ('sharpe_ratio',0):.2f} |\n"

        # Puis les benchmarks
        for bench_name,bench_metrics in comparison.items ():
            summary+=f"| {bench_name} | {bench_metrics.get ('total_return',0):.2f}% | {bench_metrics.get ('max_drawdown',0):.2f}% | {bench_metrics.get ('sharpe_ratio',0):.2f} |\n"

    # Analyse des transactions
    if 'transaction_history' in results:
        transactions=results['transaction_history']
        total_fees=sum (t.get ('fee',0) for t in transactions)
        avg_fee=total_fees / len (transactions) if transactions else 0

        summary+="\n## Statistiques des Transactions\n\n"
        summary+=f"- Nombre total de transactions: {len (transactions)}\n"
        summary+=f"- Frais totaux: ${total_fees:.2f}\n"
        summary+=f"- Frais moyen par transaction: ${avg_fee:.2f}\n"
        summary+=f"- Impact des frais sur la performance: {results.get ('fee_drag',0):.2f}%\n"

    # Sauvegarde si chemin spécifié
    if file_path:
        try:
            with open (file_path,'w') as f:
                f.write (summary)
            logger.info (f"Résumé sauvegardé à {file_path}")
        except Exception as e:
            logger.warning (f"Échec de la sauvegarde du résumé: {str (e)}")

    return summary
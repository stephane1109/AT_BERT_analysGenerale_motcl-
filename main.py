import os
import math
import re
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import networkx as nx
from pyvis.network import Network
from collections import Counter
import community.community_louvain as community_louvain
import spacy

# Configuration spaCy
spacy_nlp = spacy.load("fr_core_news_sm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paramètres
MAX_LENGTH = 512
NB_NEIGHBORS_DEFAULT = 20
TOP_N_DEFAULT = 100

MODEL_NAME = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Variables globales
selected_filepath = ""
terme_central_value = ""
var_debug = True
text_widget = None

# Variables Tkinter
pooling_method = None
var_intra = None
var_stopwords = None
var_lemmatisation = None
analysis_mode = None

def afficher_message(msg):
    global text_widget
    if text_widget is not None:
        text_widget.insert(tk.END, msg + "\n")
        text_widget.see(tk.END)
    else:
        print(msg)

def selectionner_fichier():
    global selected_filepath
    path = filedialog.askopenfilename(filetypes=[("Fichiers texte", "*.txt"), ("Tous", "*.*")])
    if path:
        selected_filepath = path
        afficher_message("Fichier sélectionné : " + selected_filepath)

# ----------------------
# Prétraitement
# ----------------------

def pretraiter_phrase(phrase):
    doc = spacy_nlp(phrase)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"]]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        else:
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"]]
    return " ".join(tokens)

def extraire_termes_frequents(texte, top_n):
    doc = spacy_nlp(texte)
    if var_lemmatisation.get():
        if var_stopwords.get():
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.lemma_.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    else:
        if var_stopwords.get():
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) >= 4]
        else:
            tokens = [token.text.lower() for token in doc
                      if token.pos_ in ["NOUN", "PROPN"] and len(token.text) >= 4]
    freq = Counter(tokens)
    return dict(freq.most_common(top_n))

def normaliser_texte(text):
    lignes = text.splitlines()
    lignes_filtrees = [l for l in lignes if not l.strip().startswith("****")]
    texte_filtre = " ".join(lignes_filtrees)
    return re.sub(r'\s+', ' ', texte_filtre).strip().lower()

def split_text_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def decouper_texte_en_segments(text, tokenizer, max_length, overlap):
    sentences = re.split(r'([.!?])', text)
    segments = []
    current_segment = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if i+1 < len(sentences):
            sentence += sentences[i+1]
        candidate = (current_segment + " " + sentence).strip() if current_segment else sentence
        tokens_candidate = tokenizer.encode(candidate, add_special_tokens=True)
        if len(tokens_candidate) > max_length:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence
        else:
            current_segment = candidate
    if current_segment:
        segments.append(current_segment.strip())
    return segments

# ----------------------
# Embedding
# ----------------------

def encoder_phrase(phrase):
    from collections import Counter
    phrase_pretraitee = pretraiter_phrase(phrase)
    inputs = tokenizer(phrase_pretraitee, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    method = pooling_method.get()
    if method == "mean":
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)
    elif method == "weighted":
        tokens_pretraite = phrase_pretraitee.split()
        freq_dict = Counter(tokens_pretraite)
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                word = token[1:]
                weights.append(freq_dict.get(word, 1))
            else:
                weights.append(weights[-1] if weights else 1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        weights_tensor = weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * weights_tensor).sum(dim=1)
        normalization = weights_tensor.sum()
        emb = weighted_embeds / normalization
        return emb.cpu().numpy().squeeze(0)
    elif method == "max":
        emb = outputs.last_hidden_state.max(dim=1)[0]
        return emb.cpu().numpy().squeeze(0)
    elif method == "attention":
        token_embeds = outputs.last_hidden_state
        norms = torch.norm(token_embeds, dim=2)
        att_weights = torch.softmax(norms, dim=1)
        weighted_embeds = (token_embeds * att_weights.unsqueeze(2)).sum(dim=1)
        return weighted_embeds.cpu().numpy().squeeze(0)
    elif method == "sif":
        a = 0.001
        tokens_pretraite = phrase_pretraitee.split()
        if len(tokens_pretraite) == 0:
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)
        from collections import Counter
        counts = Counter(tokens_pretraite)
        total_tokens = len(tokens_pretraite)
        sif_weights_list = [a / (a + counts[token] / total_tokens) for token in tokens_pretraite]
        tokens_ids = inputs['input_ids'][0]
        tokens_from_ids = tokenizer.convert_ids_to_tokens(tokens_ids)
        if tokens_from_ids[0] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[1:]
            outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
        if tokens_from_ids[-1] in tokenizer.all_special_tokens:
            tokens_from_ids = tokens_from_ids[:-1]
            outputs.last_hidden_state = outputs.last_hidden_state[:, :-1, :]
        sif_weights = []
        for token in tokens_from_ids:
            if token.startswith("▁"):
                sif_weights.append(sif_weights_list.pop(0) if sif_weights_list else 1)
            else:
                sif_weights.append(sif_weights[-1] if sif_weights else 1)
        sif_weights_tensor = torch.tensor(sif_weights, dtype=torch.float32, device=outputs.last_hidden_state.device)
        sif_weights_tensor = sif_weights_tensor.unsqueeze(0).unsqueeze(-1)
        weighted_embeds = (outputs.last_hidden_state * sif_weights_tensor).sum(dim=1)
        normalization = sif_weights_tensor.sum()
        emb = weighted_embeds / normalization
        return emb.cpu().numpy().squeeze(0)
    else:
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().squeeze(0)

def encoder_contextuel_simplifie(texte, mot_cle):
    sentences = split_text_into_sentences(texte)
    pertinentes = [s for s in sentences if mot_cle.lower() in s.lower()]
    afficher_message(f"Nombre de phrases contextuelles pour '{mot_cle}' : {len(pertinentes)}")
    if not pertinentes:
        return encoder_phrase(mot_cle)
    embeddings = [encoder_phrase(s) for s in pertinentes]
    if pooling_method.get() == "sif":
        X = np.vstack(embeddings)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pc = Vt[0]
        embeddings = [emb - np.dot(emb, pc) * pc for emb in embeddings]
    return np.mean(embeddings, axis=0)

def encoder_terme_par_contexte(terme, texte):
    sentences = split_text_into_sentences(texte)
    context_sentences = [s for s in sentences if terme.lower() in s.lower()]
    if not context_sentences:
        return encoder_phrase(terme), []
    embeddings = [encoder_phrase(s) for s in context_sentences]
    if pooling_method.get() == "sif":
        X = np.vstack(embeddings)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pc = Vt[0]
        embeddings = [emb - np.dot(emb, pc) * pc for emb in embeddings]
    return np.mean(embeddings, axis=0), context_sentences

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ----------------------
# Construction de graphes
# ----------------------

def construire_voisins_contextuel(texte, mot_cle, embedding_keyword):
    top_n = int(entry_nb_termes.get())
    freq_dict = extraire_termes_frequents(texte, top_n)
    candidats = list(freq_dict.keys())
    afficher_message(f"DEBUG: {len(candidats)} candidats fréquents extraits du corpus.")
    voisins = []
    cache = {}
    for idx, t in enumerate(candidats):
        if t.lower() == mot_cle.lower():
            continue
        if t in cache:
            emb_t, passages = cache[t]
        else:
            emb_t, passages = encoder_terme_par_contexte(t, texte)
            cache[t] = (emb_t, passages)
        sim = cosine_similarity(embedding_keyword, emb_t)
        afficher_message(f"Candidat : {t} - Similarité : {sim:.4f}")
        voisins.append((t, sim, passages))
        progress_bar['maximum'] = len(candidats)
        progress_bar['value'] = idx + 1
    voisins = sorted(voisins, key=lambda x: x[1], reverse=True)[:int(entry_voisins.get())]
    return voisins, freq_dict, cache

def construire_graphe_general(freq_dict, embeddings_cache, threshold):
    G = nx.Graph()
    for term in freq_dict.keys():
        G.add_node(term, frequency=freq_dict.get(term, "N/A"))
    terms = list(freq_dict.keys())
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            sim = cosine_similarity(embeddings_cache[terms[i]], embeddings_cache[terms[j]])
            if sim >= threshold:
                G.add_edge(terms[i], terms[j], weight=sim)
    return G

def colorer_communautes(G, freq_dict):
    partition = community_louvain.best_partition(G)
    palette = ["#8A2BE2", "#9370DB", "#BA55D3", "#DA70D6", "#D8BFD8"]
    for node in G.nodes():
        comm = partition.get(node, 0)
        color = palette[comm % len(palette)]
        freq = freq_dict.get(node, "N/A")
        G.nodes[node]["frequency"] = freq
        G.nodes[node]["color"] = color
    return G, partition

def assigner_layout_etoile(G, central, centre=(300,300), rayon=300):
    positions = {central: {"x": centre[0], "y": centre[1]}}
    voisins = list(G.neighbors(central))
    n = len(voisins)
    if n > 0:
        angle_gap = 2*math.pi / n
        for i, node in enumerate(voisins):
            angle = i * angle_gap
            x = centre[0] + rayon * math.cos(angle)
            y = centre[1] + rayon * math.sin(angle)
            positions[node] = {"x": x, "y": y}
    return positions

def assigner_layout_classique(G):
    positions = nx.spring_layout(G, seed=42)
    positions_dict = {}
    for node, coord in positions.items():
        positions_dict[node] = {"x": float(coord[0]*1000), "y": float(coord[1]*1000)}
    return positions_dict

def nx_vers_pyvis_motcle(G, positions, central):
    """
    Graphe circulaire pour l'analyse par mot-clé :
      - Mot-clé (central) en blanc
      - Autres nœuds en violet
      - Arêtes en blanc
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options('{ "physics": { "enabled": false } }')

    for node in G.nodes():
        freq = G.nodes[node].get("frequency", 0)
        if node.lower() == central.lower():
            color = "#FFFFFF"  # blanc pour le mot-clé
        else:
            color = "#8A2BE2"  # violet
        label = f"{node}\nFreq: {freq}"
        if isinstance(freq, (int, float)):
            size = 20 + freq*2
        else:
            size = 20
        pos = positions.get(node, {"x":300, "y":300})
        net.add_node(node, label=label, title=label, x=pos["x"], y=pos["y"],
                     color=color, size=size)

    # Arêtes blanches
    for u, v, data in G.edges(data=True):
        sim = data.get("weight", 0)
        net.add_edge(u, v, title=f"Sim: {sim:.4f}", color="#FFFFFF")

    return net

def nx_vers_pyvis_general(G, positions):
    """
    Graphique pour l'analyse générale :
      - Couleurs héritées de colorer_communautes
      - Layout spring
    """
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options('{ "physics": { "enabled": false } }')

    for node, data in G.nodes(data=True):
        freq = data.get("frequency", 0)
        color = data.get("color", "#8A2BE2")
        label = f"{node}\nFreq: {freq}"
        if isinstance(freq, (int, float)):
            size = 20 + freq*2
        else:
            size = 20
        pos = positions.get(node, {"x":300, "y":300})
        net.add_node(node, label=label, title=label,
                     x=pos["x"], y=pos["y"], color=color, size=size)

    for u, v, data in G.edges(data=True):
        weight = float(data.get("weight", 0))
        color = data.get("color", "#FFFFFF")
        net.add_edge(u, v, value=weight, title=f"Sim: {weight:.4f}", color=color)
    return net

def sauvegarder_resultats(voisins, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for mot, sim, passages in voisins:
            f.write(f"{mot}\t{sim:.4f}\n")
            if passages:
                for passage in passages:
                    f.write(f"    {passage}\n")
            f.write("\n")

def creer_graphe(voisins, mot_cle, freq_dict):
    G = nx.Graph()
    G.add_node(mot_cle, frequency=freq_dict.get(mot_cle, "N/A"))
    for voisin, sim, _ in voisins:
        if voisin.lower() == mot_cle.lower():
            continue
        G.add_node(voisin, frequency=freq_dict.get(voisin, "N/A"))
        G.add_edge(mot_cle, voisin, weight=sim)
    return G

def ajouter_aretes_intercommunautaires_motcle(G, central):
    """
    Méthode intracommunautaire spéciale pour le mode mot-clé.
    Par exemple, on peut relier les hubs de chaque communauté
    par une unique arête grise, sauf la communauté du mot-clé central.
    """
    # ... code similaire à 'ajouter_aretes_intercommunautaires' ...
    # Personnalisez si nécessaire
    return G

def ajouter_aretes_intercommunautaires_general(G, embeddings_cache):
    """
    Méthode intracommunautaire pour l'analyse générale,
    éventuellement un autre critère de sélection.
    """
    # ... un code spécifique pour le mode général ...
    return G

# ----------------------------------------------------------------------------
# Analyser par mot clé
# ----------------------------------------------------------------------------

def analyser_fichier_mot_cle(texte):
    mot_cle = entry_noeud_central.get().strip().lower()
    afficher_message(f"Analyse du mot-clé : {mot_cle}")

    embedding_keyword = encoder_contextuel_simplifie(texte, mot_cle)
    norm_keyword = np.linalg.norm(embedding_keyword)
    afficher_message(f"Embedding du mot-clé (norme) : {norm_keyword:.4f}")

    voisins, freq_dict, cache = construire_voisins_contextuel(texte, mot_cle, embedding_keyword)
    try:
        nb_voisins = int(entry_voisins.get())
    except:
        nb_voisins = NB_NEIGHBORS_DEFAULT
    voisins = voisins[:nb_voisins]
    afficher_message(f"{len(voisins)} voisins positifs sélectionnés.")
    sauvegarder_resultats(voisins, "context_neighbors.txt")
    afficher_message("Fichier texte généré : context_neighbors.txt")

    G = creer_graphe(voisins, mot_cle, freq_dict)
    # Colorer (optionnel)
    G, partition = colorer_communautes(G, freq_dict)

    # Arêtes intracommunautaires
    if var_intra.get():
        # On peut appeler la méthode spéciale mot-clé
        G = ajouter_aretes_intercommunautaires_motcle(G, mot_cle)
        afficher_message("Arêtes intracommunautaires mot-clé ajoutées.")

    # Layout en étoile
    positions = assigner_layout_etoile(G, mot_cle)
    # Conversion Pyvis mot-clé
    net = nx_vers_pyvis_motcle(G, positions, mot_cle)

    html_file = "graph.html"
    net.write_html(html_file)
    afficher_message("Graphique généré : " + os.path.abspath(html_file))

# ----------------------------------------------------------------------------
# Analyser mode général
# ----------------------------------------------------------------------------

def analyser_fichier_general(texte):
    afficher_message("Analyse générale du corpus")
    top_n = int(entry_nb_termes.get())
    freq_dict = extraire_termes_frequents(texte, top_n)
    embeddings_cache = {}
    for term in freq_dict.keys():
        emb, _ = encoder_terme_par_contexte(term, texte)
        embeddings_cache[term] = emb

    # Choix du mode d'affichage des arêtes
    if edge_mode.get() == "mst":
        threshold = float(entry_threshold_high.get())
        G_complete = construire_graphe_general(freq_dict, embeddings_cache, threshold)
        from networkx.algorithms.tree import maximum_spanning_tree
        G_final = maximum_spanning_tree(G_complete)
    elif edge_mode.get() == "knn":
        k = int(entry_knn.get())
        G_final = construire_graphe_knn(freq_dict, embeddings_cache, k)
    else:
        G_final = construire_graphe_general(freq_dict, embeddings_cache, float(entry_threshold_high.get()))

    G_final, partition = colorer_communautes(G_final, freq_dict)

    if var_intra.get():
        # On peut appeler la méthode spéciale générale
        G_final = ajouter_aretes_intercommunautaires_general(G_final, embeddings_cache)
        afficher_message("Arêtes intracommunautaires général ajoutées.")

    positions = assigner_layout_classique(G_final)
    net = nx_vers_pyvis_general(G_final, positions)
    html_file = "graph_general.html"
    net.write_html(html_file)
    afficher_message("Graphique général généré : " + os.path.abspath(html_file))

def construire_graphe_knn(freq_dict, embeddings_cache, k):
    G = nx.Graph()
    for term in freq_dict.keys():
        G.add_node(term, frequency=freq_dict.get(term, "N/A"))
    terms = list(freq_dict.keys())
    for term in terms:
        similarities = []
        for other in terms:
            if other == term:
                continue
            sim = cosine_similarity(embeddings_cache[term], embeddings_cache[other])
            similarities.append((other, sim))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        for other, sim in similarities:
            G.add_edge(term, other, weight=sim)
    return G

# ----------------------------------------------------------------------------
# Bouton principal
# ----------------------------------------------------------------------------

def analyser_fichier():
    global selected_filepath, terme_central_value
    if not selected_filepath:
        afficher_message("Erreur : veuillez sélectionner un fichier.")
        return
    with open(selected_filepath, "r", encoding="utf-8") as f:
        texte_brut = f.read().strip()
    texte_normalise = normaliser_texte(texte_brut)
    texte_complet = texte_normalise

    if analysis_mode.get() == "mot_cle":
        analyser_fichier_mot_cle(texte_complet)
    else:
        analyser_fichier_general(texte_complet)

# ----------------------------------------------------------------------------
# Interface Tkinter
# ----------------------------------------------------------------------------

root = tk.Tk()
root.title("Analyse textuelle – Choix du mode d'analyse")
root.geometry("800x800")

analysis_mode = tk.StringVar(value="mot_cle")
frame_mode = ttk.LabelFrame(root, text="Mode d'analyse", padding="10")
frame_mode.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
ttk.Radiobutton(frame_mode, text="Analyse par mot-clé", variable=analysis_mode, value="mot_cle").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_mode, text="Analyse générale", variable=analysis_mode, value="general").grid(row=0, column=1, sticky=tk.W)

frame_params = ttk.LabelFrame(root, text="Paramètres", padding="10")
frame_params.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
ttk.Button(frame_params, text="Sélectionner un fichier", command=selectionner_fichier).grid(row=0, column=0, columnspan=2, pady=5)
ttk.Label(frame_params, text="Nombre de voisins/termes (max) :").grid(row=1, column=0, sticky=tk.W)
entry_voisins = ttk.Entry(frame_params, width=10)
entry_voisins.insert(0, "20")
entry_voisins.grid(row=1, column=1, sticky=tk.W)
ttk.Label(frame_params, text="Nombre de termes à analyser :").grid(row=2, column=0, sticky=tk.W)
entry_nb_termes = ttk.Entry(frame_params, width=10)
entry_nb_termes.insert(0, "100")
entry_nb_termes.grid(row=2, column=1, sticky=tk.W)
ttk.Label(frame_params, text="Seuil similarité (entre 0 et 1) :").grid(row=3, column=0, sticky=tk.W)
entry_threshold_high = ttk.Entry(frame_params, width=10)
entry_threshold_high.insert(0, "0.8")
entry_threshold_high.grid(row=3, column=1, sticky=tk.W)

frame_pretraitement = ttk.LabelFrame(root, text="Options de prétraitement", padding="10")
frame_pretraitement.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
var_stopwords = tk.BooleanVar(value=True)
var_lemmatisation = tk.BooleanVar(value=True)
ttk.Checkbutton(frame_pretraitement, text="Utiliser stopwords", variable=var_stopwords).grid(row=0, column=0, sticky=tk.W)
ttk.Checkbutton(frame_pretraitement, text="Utiliser lemmatisation", variable=var_lemmatisation).grid(row=0, column=1, sticky=tk.W)

frame_embedding = ttk.LabelFrame(root, text="Méthode d'embedding", padding="10")
frame_embedding.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
pooling_method = tk.StringVar(value="mean")
ttk.Radiobutton(frame_embedding, text="Mean pooling", variable=pooling_method, value="mean").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Weighted pooling (fréquence)", variable=pooling_method, value="weighted").grid(row=0, column=1, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Max pooling", variable=pooling_method, value="max").grid(row=0, column=2, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="Attention pooling", variable=pooling_method, value="attention").grid(row=0, column=3, sticky=tk.W)
ttk.Radiobutton(frame_embedding, text="SIF pooling", variable=pooling_method, value="sif").grid(row=0, column=4, sticky=tk.W)

frame_edge = ttk.LabelFrame(root, text="Mode d'affichage des arêtes (pour analyse générale)", padding="10")
frame_edge.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
edge_mode = tk.StringVar(value="mst")
ttk.Radiobutton(frame_edge, text="Arbre couvrant maximal (MST)", variable=edge_mode, value="mst").grid(row=0, column=0, sticky=tk.W)
ttk.Radiobutton(frame_edge, text="Graphe k-NN", variable=edge_mode, value="knn").grid(row=0, column=1, sticky=tk.W)
ttk.Label(frame_edge, text="Nombre k (pour k-NN) :").grid(row=1, column=0, sticky=tk.W)
entry_knn = ttk.Entry(frame_edge, width=10)
entry_knn.insert(0, "1")
entry_knn.grid(row=1, column=1, sticky=tk.W)

frame_analysis = ttk.LabelFrame(root, text="Mot-clé (pour analyse par mot-clé)", padding="10")
frame_analysis.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W+tk.E)
entry_noeud_central = ttk.Entry(frame_analysis, width=20)
entry_noeud_central.insert(0, "soins")
entry_noeud_central.grid(row=0, column=1, sticky=tk.W)
ttk.Label(frame_analysis, text="(Ce champ sera ignoré en mode général)").grid(row=0, column=2, sticky=tk.W)
var_intra = tk.BooleanVar(value=False)
ttk.Checkbutton(frame_analysis, text="Ajouter arêtes intracommunautaires", variable=var_intra).grid(row=1, column=0, columnspan=2, sticky=tk.W)

ttk.Button(root, text="Lancer l'analyse", command=analyser_fichier).grid(row=6, column=0, pady=10)
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate")
progress_bar.grid(row=7, column=0, padx=10, pady=10, sticky="ew")

text_widget = scrolledtext.ScrolledText(root, width=80, height=20)
text_widget.grid(row=8, column=0, padx=10, pady=10)

root.mainloop()

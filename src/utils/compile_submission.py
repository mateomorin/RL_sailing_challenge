import base64
import zlib
import re
import os
import numpy as np
import zipfile

def create_standalone_submission(script_path, weights_path, output_path):
    print(f"Lecture des poids depuis {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Erreur : Le fichier de poids {weights_path} est introuvable.")
        return
        
    with open(weights_path, "rb") as f:
        raw_weights = f.read()
        
    # Compression et encodage en Base64
    compressed_weights = zlib.compress(raw_weights)
    b64_weights = base64.b64encode(compressed_weights).decode('utf-8')
    print(f"Poids compressés et encodés. Taille : {len(b64_weights)/1024:.2f} KB")

    print(f"Lecture du script agent_massive_training.py...")
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Code de remplacement pour la méthode load
    embedded_load_code = """
    def load(self, path: str) -> None:
        \"\"\"Charge les poids directement depuis la chaîne de caractères Base64 embarquée.\"\"\"
        import base64
        import zlib
        import io
        try:
            compressed_bytes = base64.b64decode(WEIGHTS_B64)
            npz_bytes = zlib.decompress(compressed_bytes)
            with io.BytesIO(npz_bytes) as f:
                data = np.load(f, allow_pickle=True)
                weights = {k: data[k] for k in data.files if not k.startswith('_')}
            self._net = NumpyActorCritic(weights)
            print("[MyAgent] Succès : Poids chargés depuis la mémoire (Base64) !")
        except Exception as e:
            print(f"[MyAgent] Erreur critique : {e}")
            self._net = None
"""

    # Injection du nouveau code de chargement
    pattern = r"def load\(self, path: str\) -> None:.*?self\._net = None"
    modified_content = re.sub(pattern, embedded_load_code.strip(), content, flags=re.DOTALL)
    
    # Ajout de la constante à la fin du fichier
    final_code = modified_content + "\n\n# --- POIDS EMBARQUÉS ---\n"
    final_code += f"WEIGHTS_B64 = \"\"\"{b64_weights}\"\"\"\n"

    # Write the submission zip with the .py file at the root
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("my_agent.py", final_code)

    print(f"Created {output_path}")
    print("Contents:")
    with zipfile.ZipFile(output_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")

    print(f"✓ Terminé ! Fichier créé : {output_path}")

if __name__ == '__main__':
    # Modifiez ici si vos fichiers ont des noms différents
    create_standalone_submission(
        script_path='src/agents/ppo_bc_submission.py', 
        weights_path='bc_10_epoch_finetuned.npz', 
        output_path='bc_10_ft.zip'
    )
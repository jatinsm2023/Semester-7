from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict
import json
import hashlib
import io
from typing import List
from flask import Flask, jsonify, render_template, request, send_file, abort
import yaml

def get_time():
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class block:
    index: int
    timestamp: str
    data: str
    prev_hash: str
    nonce: int = 0
    hash: str = ""

    def get_hash(self):
        data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "nonce": self.nonce,
        }
        dumped_data = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped_data.encode("utf-8")).hexdigest()


class blockchain:
    def __init__(self, difficulty: int = 4):
        self.difficulty = difficulty
        self.chain: List[block] = []
        self.create_first_block()

    def create_first_block(self):
        first_block = block(
            index = 0, 
            timestamp = get_time(),
            data = "Genesis Block",
            prev_hash = "0",
            nonce = 0
        )
        self.mine(first_block)
        self.chain.append(first_block)

    def mine(self, block):
        prefix = "0" * self.difficulty
        while True: 
            hsh = block.get_hash()
            if hsh.startswith(prefix):
                block.hash = hsh
                return
            block.nonce += 1

    def add_new_block(self, data):
        prev = self.chain[-1]
        new_block = block(
            index = prev.index + 1,
            timestamp = get_time(),
            data = data,
            prev_hash = prev.hash,
            nonce = 0,
        )
        self.mine(new_block)
        self.chain.append(new_block)
        return new_block 


    @staticmethod
    def validate_blockchain(chain: List[block], difficulty: int):
        prefix = "0"*difficulty

        if not chain:
            return False, 0
        g = chain[0]

        if g.prev_hash != "0":
            return False, 0
        
        if g.get_hash() != g.hash or not g.hash.startswith(prefix):
            return False, 0

        for i in range(1, len(chain)):
            curr = chain[i]
            prev = chain[i-1]
            if curr.prev_hash != prev.hash:
                return False, i
            
            if curr.get_hash() != curr.hash or not curr.hash.startswith(prefix):
                return False, i
            
        return True, None

    def to_dict_list(self):
        return [asdict(b) for b in self.chain]

    @staticmethod
    def from_dict_list(items):
        return [block(**b) for b in items]


app = Flask(__name__)
global_blockchain = blockchain(difficulty=4)

@app.route("/")
def index():
    return render_template("index.html", difficulty=global_blockchain.difficulty)

@app.get("/chain")
def get_chain():
    return jsonify({
        "difficulty": global_blockchain.difficulty,
        "length": len(global_blockchain.chain),
        "chain": global_blockchain.to_dict_list()
    })

@app.post("/mine")
def mine():
    paylaod = request.get_json(silent=True) or {}
    data = paylaod.get("data")
    block = global_blockchain.add_new_block(data)
    return jsonify({
        "ok": True, 
        "block": asdict(block),
        "length": len(global_blockchain.chain)
    })

@app.get("/download")
def download_chain():
    filetype = (request.args.get("format") or "json").lower()
    data = global_blockchain.to_dict_list()

    if filetype == "json":
        buf = io.BytesIO(json.dumps(data,indent=2).encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True, download_name="blockchain.json")

    if filetype == "yaml":
        buf = io.BytesIO(yaml.safe_dump(data, sort_keys=False).encode("utf-8"))
        return send_file(buf, mimetype="application/x-yaml", as_attachment=True, download_name="blockchain.yaml")

    if filetype == "txt":
        lines = "\n".join(json.dumps(b, separators=(",", ":")) for b in data)
        buf = io.BytesIO(lines.encode("utf-8"))
        return send_file(buf, mimetype="text/plain", as_attachment=True, download_name="blockchain.txt")

    abort(400, "Unsupported format. Use one of: json|yaml|txt")

@app.post("/validate")
def validate_upload():
    if "file" not in request.files:
        abort(400, "No file uploaded")

    f = request.files["file"]
    content = f.read().decode("utf-8", errors="replace")
    name = (f.filename or "").lower()

    parsed: List[Dict[str, Any]] = []

    try:
        if name.endswith(".json"):
            parsed = json.loads(content)
        elif name.endswith(".yaml") or name.endswith(".yml"):
            parsed = yaml.safe_load(content)
        elif name.endswith(".txt"):
            parsed = [json.loads(line) for line in content.strip().splitlines() if line.strip()]
        else:
            try:
                parsed = json.loads(content)
            except Exception:
                try:
                    parsed = yaml.safe_load(content)
                except Exception:
                    parsed = [json.loads(line) for line in content.strip().splitlines() if line.strip()]

    except Exception as e:
        abort(400, f"Could not parse uploaded file: {e}")

    try:
        blocks = global_blockchain.from_dict_list(parsed)
    except Exception as e:
        abort(400, f"Invalid block structure: {e}")

    ok, invalid_from = global_blockchain.validate_blockchain(blocks, difficulty=global_blockchain.difficulty)

    return jsonify({
        "ok": ok,
        "invalid_from": invalid_from,  
        "length": len(blocks),
        "chain": [asdict(b) for b in blocks],
        "difficulty": global_blockchain.difficulty
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

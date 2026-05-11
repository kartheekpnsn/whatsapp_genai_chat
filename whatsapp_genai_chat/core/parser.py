from pathlib import Path
import pandas as pd
from whatsappchattodf import WhatsappChatToDF


def parse_chat(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chat file not found: {path}")
    chat = WhatsappChatToDF(str(path))
    df = chat.run()
    df = df.rename(columns={df.columns[1]: "sender", df.columns[2]: "message"})
    df = df[df["sender"].notna() & df["message"].notna()].reset_index(drop=True)
    return df

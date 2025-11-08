# quick_rollout.py
from ofc_env import OFCEnv
from dqn_ofc import run_test, train

if __name__ == "__main__":
    # まずは学習 100 エピソードだけ走らせて動作確認
    train(num_episodes=100, resume=False, save_path="./ckpt/ofc_qnet.pt")
    # 学習済みで試行
    run_test(model_path="./ckpt/ofc_qnet.pt")

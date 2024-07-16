class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer
        # answerの辞書を作成
        self.answer2idx = {}
        self.idx2answer = {}
        if self.answer:
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)
    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．
        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer
    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．
        Parameters
        ----------
        idx : int
            取得するデータのインデックス
        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : str
            質問文
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = self.df["question"][idx]
        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）
            return image, question, torch.Tensor(answers), int(mode_answer_idx)
        else:
            return image, question
    def __len__(self):
        return len(self.df)
# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.
    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10
    return total_acc / len(batch_pred)
# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)
    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])
class VQAModel(nn.Module):
    def __init__(self, bert_model_name: str, n_answer: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.resnet = ResNet18()
        self.fc = nn.Sequential(
            nn.Linear(768 + 512, 512),  # BERTの出力サイズは768
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropoutを追加
            nn.Linear(512, n_answer)
        )
    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        # 質問をトークン化し、BERTに入力
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(image.device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        question_feature = outputs.last_hidden_state[:, 0, :]  # [CLS]トークンの出力を使用
        # 画像特徴量と質問特徴量を結合
        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)
        return x
def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    for image, question, answers, mode_answer in dataloader:
        image, answers, mode_answer = \
            image.to(device), answers.to(device), mode_answer.to(device)
        optimizer.zero_grad()
        with autocast():
            pred = model(image, question)
            loss = criterion(pred, mode_answer.squeeze())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start
def eval(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    simple_acc = 0
    start = time.time()
    with torch.no_grad():
        for image, question, answers, mode_answer in dataloader:
            image, answers, mode_answer = \
                image.to(device), answers.to(device), mode_answer.to(device)
            with autocast():
                pred = model(image, question)
                loss = criterion(pred, mode_answer.squeeze())
            total_loss += loss.item()
            total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
            simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy
    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start
def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CuDNNの設定
    torch.backends.cudnn.benchmark = True
    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="/content/drive/MyDrive/DL/last/VQA/train.json", image_dir="/content/drive/MyDrive/DL/last/train", transform=transform)
    test_dataset = VQADataset(df_path="/content/drive/MyDrive/DL/last/VQA/valid.json", image_dir="/content/drive/MyDrive/DL/last/valid", transform=transform, answer=False)
    test_dataset.update_dict(train_dataset)
    # DataLoaderの設定
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )
    model = VQAModel(bert_model_name='bert-base-uncased', n_answer=len(train_dataset.answer2idx)).to(device)
    # optimizer / criterion
    num_epoch = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scaler = GradScaler()
    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device, scaler)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        with autocast():
            pred = model(image, question)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)
    submission = [train_dataset.idx2answer[id] for id in submission]
    submission = np.array(submission)
    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", submission)
if __name__ == "__main__":
    main()
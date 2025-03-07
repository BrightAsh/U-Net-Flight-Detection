% 추출할 데이터 선택 (데이터 index는 시뮬레이션 비행데이터 종류.xlsx 참조)

label = 50;

% 3  q 6 theta 18 alpha

data_index = [3, 6, 18]; 
training_data_path = 'D:/deeplearning/Unet_final/data preprocessing/1 matlab data/training_data/종';
test_data_path = 'D:/deeplearning/Unet_final/data preprocessing/1 matlab data/test_data/종';

mkdir(training_data_path)
mkdir(test_data_path)

sampling_term = 5;
training_path = 'D:/deeplearning/Upset_sim_data/Parsing_data(150Hz)/Train_data/150Hz';

normal_flight_range = 701;

[flight_data_max, flight_data_min] = find_max_min_forFlightUpsetData(training_path, data_index, label,normal_flight_range);

for initv = 15:3:21
    % value prediction을 위해서 cmdv = 15 (모든 데이터 정상 상태)는 제외
    for cmdv = 5:5:15
        for cmdPhi = -45:15:45
            % .mat file raw data 불러오기
            path = strcat(training_path,'/L_origin_H300_initV',num2str(initv),'_cmdV',num2str(cmdv),'_initPhi0_cmdPhi',num2str(cmdPhi),'_150Hz_Cut.mat');
            load(path);
            parsing_data = origin_data;
            if cmdv == 15
                zero_index = normal_flight_range;
            else
                zero_index = size(parsing_data,1);
            end
            for start_index = 1:sampling_term
                normalized_data = [];
                for j = 1:length(data_index) 
                    normalized_data = [normalized_data (parsing_data(start_index:sampling_term:zero_index, data_index(j))-flight_data_min(j)) / (flight_data_max(j) - flight_data_min(j))]; 
                end
                path = strcat(training_data_path, '/L_origin_H300_initV',num2str(initv),'_cmdV',num2str(cmdv),'_initPhi0_cmdPhi',num2str(cmdPhi),'_',num2str(start_index),'_30Hz.csv');
                normalized_data = [normalized_data parsing_data(start_index:sampling_term:zero_index, label)];
                writematrix(normalized_data, path);
            end
        end
    end
end

% origin test data 경로
test_dataPath = [];
root_path_not = 'D:/deeplearning/Upset_sim_data/Parsing_data(150Hz)/Test_data/150Hz';
root_path_used = 'D:/deeplearning/Upset_sim_data/Parsing_data(150Hz)/Train_data/150Hz';


% UT-1
path = root_path_not +  "/L_origin_H300_initV17_cmdV13_initPhi0_cmdPhi-32_150Hz_Cut.mat";
test_dataPath = [test_dataPath path];

% UT-2
path = root_path_not +  "/T-2_long_150Hz.mat";
test_dataPath = [test_dataPath path];

% UT-3
path = root_path_not + "/L_origin_H300_initV20_cmdV11_initPhi0_cmdPhi32_150Hz_Cut.mat";
test_dataPath = [test_dataPath path];


% used_condition
path = root_path_not + "/L_origin_H300_initV15_cmdV10_initPhi0_cmdPhi30_150Hz_Long.csv";
test_dataPath = [test_dataPath path];


for i = 1:length(test_dataPath)
    normalized_data = [];
    if i == 4
        data = readtable(test_dataPath(i));
        origin_data = table2array(data);
    
    else
        load(test_dataPath(i));
    end

    zero_index =size(origin_data,1);

    for j = 1:length(data_index)
        normalized_data = [normalized_data (origin_data(1:sampling_term:zero_index, data_index(j))-flight_data_min(j)) / (flight_data_max(j) - flight_data_min(j))];
    end
    normalized_data = [normalized_data origin_data(1:sampling_term:zero_index, label)];

    %==========================================================================================================
    path_split = split(test_dataPath(i),["/", "."]);
    path = strcat(test_data_path, '/', path_split(length(path_split)-1), '.csv');

    % 정규화된 테스트 데이터 저장
    writematrix(normalized_data, path);
end
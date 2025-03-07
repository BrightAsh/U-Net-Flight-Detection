% 옳바른 정규화 방식은 Training data set에서만 찾은 max-min값으로 정규화 진행
% Test data에 대해서도 Training data set에서 찾은 max-min값으로 정규화.
% Validation set은 Training set에서 선별
function [upset_data_max, upset_data_min] = find_max_min_forFlightUpsetData(data_path, target_index, label,normal_flight_range)
    data_index = target_index;

    upset_data_max = zeros(1,length(data_index)); 
    upset_data_max(:) = -999;
    upset_data_min = zeros(1,length(data_index));
    upset_data_min(:) = 999;

    tmp_max = 0;
    tmp_min = 0;
   
    for initv = 15:3:21
        for cmdv = 5:5:15
            for cmdPhi = -45:15:45

                path = strcat(data_path, '/L_origin_H300_initV',num2str(initv),'_cmdV',num2str(cmdv),'_initPhi0_cmdPhi',num2str(cmdPhi),'_150Hz_Cut.mat');
                load(path);
                parsing_data = origin_data;
                
                % Upset data감축을 위해 Upset 발생 이후 1sec 만큼의 데이터만 포함 
                % Upset 발생 지점(Label=0)을 탐색
                if cmdv == 15
                    zero_index = normal_flight_range;
                else
                    zero_index = find_zero_index(parsing_data(:,label));
                end
                
                
                for j = 1:length(data_index)
                    tmp_max = max(parsing_data(1:zero_index, data_index(j)));
                    tmp_min = min(parsing_data(1:zero_index, data_index(j)));

                    if upset_data_max(j) < tmp_max
                        upset_data_max(j) = tmp_max;
                    end

                    if upset_data_min(j) > tmp_min
                        upset_data_min(j) = tmp_min;

                    end
                end
                
            end
        end
    end
end




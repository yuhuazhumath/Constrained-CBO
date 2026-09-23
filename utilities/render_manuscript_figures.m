function manifest = render_manuscript_figures(repo_root,selection)
%RENDER_MANUSCRIPT_FIGURES Render selected paper figures from saved trajectories.
% Reference values come from the corresponding experiment configurations.
arguments
    repo_root char = fileparts(fileparts(mfilename('fullpath')))
    selection string = "all"
end

repo_root = char(java.io.File(repo_root).getCanonicalPath());
raw_dir = fullfile(repo_root,'results','raw');
figure_dir = fullfile(repo_root,'results','figures');
style = publication_plot_style();

contract = manuscript_contract(figure_dir);
assert(height(contract)==34,'Expected 34 numerical figure panels.');
assert(numel(unique(contract.IncludeFilename))==34, ...
    'Figure list contains duplicate EPS filenames.');
allowed_selections = ["all","figure2","figure4", ...
    "figure5","figure6","figure7"];
assert(isscalar(selection) && any(selection==allowed_selections), ...
    'Unknown render selection: %s',selection);
if selection=="all"
    selected_figures=[2 4 5 6 7];
else
    selected_figures=str2double(extractAfter(selection,"figure"));
end
selected_rows=ismember(contract.Figure,selected_figures);
for i=find(selected_rows).'
    assert(~isfile(contract.EPSPath(i)), ...
        'Refusing to overwrite existing EPS: %s',contract.EPSPath(i));
    assert(~isfile(contract.PNGPath(i)), ...
        'Refusing to overwrite existing preview: %s',contract.PNGPath(i));
end
figure2_files = {
    fullfile(raw_dir,'fig1_2','figure2a_circle_equal_seed12000.mat')
    fullfile(raw_dir,'fig1_2','figure2b_circle_different_seed12100.mat')
    fullfile(raw_dir,'fig1_2','figure2c_parabola_seed12200.mat')};
if ismember(2,selected_figures)
    figure2_configs = config_preliminary(12000);
    render_figure2(figure2_files,figure2_configs,contract,style);
end

family_a_files = {
    fullfile(raw_dir,'fig4_5','figure4_5_segment_seed24000.mat')
    fullfile(raw_dir,'fig4_5','figure4_5_ellipse_seed24100.mat')
    fullfile(raw_dir,'fig4_5','figure4_5_line_seed24200.mat')};
if ismember(4,selected_figures)
    render_figure4(family_a_files,contract,style);
end
if ismember(5,selected_figures)
    render_figure5(family_a_files,contract,style);
end

figure6_files = {
    fullfile(raw_dir,'fig6','figure6a_d3_case1_seed36000.mat')
    fullfile(raw_dir,'fig6','figure6b_d3_case3_seed36100.mat')
    fullfile(raw_dir,'fig6','figure6c_d3_case4_seed36200.mat')
    fullfile(raw_dir,'fig6','figure6d_d3_case5_seed36300.mat')
    fullfile(raw_dir,'fig6','figure6e_d20_case2_seed36400.mat')
    fullfile(raw_dir,'fig6','figure6f_d20_case3_seed36500.mat')
    fullfile(raw_dir,'fig6','figure6g_d20_case4_seed36600.mat')};
if ismember(6,selected_figures)
    figure6_configs = config_ackley(36000);
    render_figure6(figure6_files,figure6_configs,contract,style);
end

k_values = [2 3 8 15 56 470];
figure7_files = cell(size(k_values));
for i=1:numel(k_values)
    figure7_files{i} = fullfile(raw_dir,'fig7',sprintf( ...
        'figure7_thomson_k%d_seed%d.mat', ...
        k_values(i),47000+100*(i-1)));
end
if ismember(7,selected_figures)
    render_figure7(figure7_files,k_values,contract,style);
end

for i=find(selected_rows).'
    eps_info = dir(contract.EPSPath(i));
    png_info = dir(contract.PNGPath(i));
    assert(isscalar(eps_info) && eps_info.bytes>0, ...
        'Missing or empty EPS: %s',contract.EPSPath(i));
    assert(isscalar(png_info) && png_info.bytes>0, ...
        'Missing or empty PNG: %s',contract.PNGPath(i));
end
contract.Status(selected_rows) = "GENERATED";
manifest = contract(selected_rows,:);
fprintf('Generated %d EPS panels and %d PNG previews.\n', ...
    height(manifest),height(manifest));
end

function contract = manuscript_contract(figure_dir)
figure_number = [repmat(2,3,1);repmat(4,6,1);repmat(5,12,1); ...
    repmat(6,7,1);repmat(7,6,1)];
panel = string(["a";"b";"c"; ...
    "a";"b";"c";"d";"e";"f"; ...
    "a";"b";"c";"d";"e";"f";"g";"h";"i";"j";"k";"l"; ...
    "a";"b";"c";"d";"e";"f";"g"; ...
    "a";"b";"c";"d";"e";"f"]);
filename = string({ ...
    'UnConst_is_const.eps','UnConst_isnot_const.eps','not_sphere.eps', ...
    'fg4_e.eps','fg4_f.eps','eq1_1.eps','eq1_2.eps', ...
    'eq1_1_plane.eps','eq1_2_plane.eps', ...
    'fig5_0.eps','fig5_5.eps','fig5_50.eps','fig5_100.eps', ...
    'eq1_ellipse_0.eps','eq1_ellipse_5.eps', ...
    'eq1_ellipse_50.eps','eq1_ellipse_100.eps', ...
    'eq1_plane_0.eps','eq1_plane_5.eps', ...
    'eq1_plane_50.eps','eq1_plane_100.eps', ...
    'line_seg.eps','eq2_dim3_er_1.eps','eq2_dim3_er_2.eps', ...
    'eq2_dim3_er_3.eps','ball.eps','eq2_dim20_er_1.eps', ...
    'eq2_dim20_er_2.eps', ...
    'eq3_er_2.eps','eq3_er_3.eps','eq3_er_8.eps', ...
    'eq3_er_15.eps','eq3_er_56.eps','eq3_er_470.eps'}).';
width_fraction = [repmat(0.44,3,1);repmat(0.50,6,1); ...
    repmat(0.45,4,1);repmat(0.20,8,1);repmat(0.47,7,1); ...
    repmat(0.50,6,1)];
eps_path = strings(size(filename)); png_path = strings(size(filename));
for i=1:numel(filename)
    if figure_number(i)<=2
        group = 'fig1_2';
    elseif figure_number(i)<=5
        group = 'fig4_5';
    else
        group = sprintf('fig%d',figure_number(i));
    end
    output_dir = fullfile(figure_dir,group);
    if ~isfolder(output_dir), mkdir(output_dir); end
    eps_path(i) = string(fullfile(output_dir,filename(i)));
    png_path(i) = string(fullfile(output_dir, ...
        replace(filename(i),'.eps','.png')));
end
status = repmat("PENDING",size(filename));
contract = table(figure_number,panel,filename,width_fraction,eps_path,png_path,status, ...
    'VariableNames',{'Figure','Panel','IncludeFilename','WidthFraction', ...
    'EPSPath','PNGPath','Status'});
end

function render_figure2(input_files,configs,contract,style)
output_names = ["UnConst_is_const.eps","UnConst_isnot_const.eps","not_sphere.eps"];
expected_ids = ["figure2a_circle_equal","figure2b_circle_different", ...
    "figure2c_parabola"];
for case_index=1:3
    e = load_experiment(input_files{case_index});
    assert(string(e.id)==expected_ids(case_index),'Figure 2 file/case mismatch.');
    config = configs(case_index);
    assert(string(config.id)==expected_ids(case_index), ...
        'Figure 2 configuration/case mismatch.');
    vstar = config.problem.vstar(:);
    assert(numel(vstar)==2,'Figure 2 reference must be two-dimensional.');
    if case_index<3
        methods = {'proposed','projected-cbo','quadratic-penalty-cbo','cb2o'};
        colors = [style.colors.proposed;style.colors.projected; ...
            style.colors.penalized;style.colors.cb2o];
    else
        methods = {'proposed','quadratic-penalty-cbo','cb2o'};
        colors = [style.colors.proposed;style.colors.penalized;style.colors.cb2o];
    end
    assert(isequal(e.methods,methods),'Unexpected Figure 2 method order.');
    assert(isequal(config.methods,methods), ...
        'Unexpected Figure 2 configuration method order.');
    row = contract.IncludeFilename==output_names(case_index);
    fig = make_figure(style.canvas.figure2,style);
    cleanup = onCleanup(@()close(fig));
    ax = axes(fig); configure_axes(ax,style,style.font.figure2,true); hold(ax,'on');
    mean_lines = gobjects(1,numel(methods));
    for m=1:numel(methods)
        [steps,values,active] = collect_consensus_distance_trajectory( ...
            e,methods{m},vstar);
        curve = active_mean(values,active);
        mean_lines(m) = plot(ax,steps,curve,'Color',colors(m,:), ...
            'LineWidth',style.lines.mean_width);
    end
    xlabel(ax,'Iteration $k$','Interpreter','latex');
    ylabel(ax,'$D(v_\alpha,v^*)$','Interpreter','latex');
    xlim(ax,[steps(1) steps(end)]);
    export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
    clear cleanup
end
end

function [steps,values,active] = collect_consensus_distance_trajectory( ...
    experiment,method,vstar)
method_index = find(strcmp(experiment.methods,method),1);
assert(~isempty(method_index), ...
    'render_manuscript_figures:MissingFigure2Method', ...
    'Experiment %s has no %s result.',experiment.id,method);
runs = experiment.results(method_index,:);
first = find(~cellfun(@isempty,runs),1);
assert(~isempty(first),'Figure 2 has no completed %s runs.',method);
steps = runs{first}.trajectory.steps(:);
dimension = numel(vstar);
values = NaN(numel(steps),numel(runs));
active = false(numel(steps),numel(runs));
for repetition=1:numel(runs)
    if isempty(runs{repetition}), continue; end
    trajectory = runs{repetition}.trajectory;
    assert(isequal(trajectory.steps(:),steps), ...
        'Figure 2 trajectory step-grid mismatch.');
    consensus = trajectory.consensus;
    assert(isequal(size(consensus),[dimension,numel(steps)]), ...
        'Figure 2 consensus trajectory has unexpected dimensions.');
    difference = consensus-vstar;
    values(:,repetition) = sqrt(sum(difference.^2,1)).'/sqrt(dimension);
    active(:,repetition) = trajectory.active_mask(:);
end
values(~active) = NaN;
end

function render_figure4(input_files,contract,style)
left_names = ["fg4_e.eps","eq1_1.eps","eq1_1_plane.eps"];
right_names = ["fg4_f.eps","eq1_2.eps","eq1_2_plane.eps"];
expected_ids = ["figure4_5_segment","figure4_5_ellipse","figure4_5_line"];
methods = {'proposed','quadratic-penalty-cbo','cb2o'};
colors = [style.colors.proposed;style.colors.penalized;style.colors.cb2o];
for case_index=1:3
    e = load_experiment(input_files{case_index});
    assert(string(e.id)==expected_ids(case_index),'Figure 4 file/case mismatch.');
    assert(isequal(e.methods,methods),'Unexpected Figure 4 method order.');

    row = contract.IncludeFilename==left_names(case_index);
    fig = make_figure(style.canvas.figure4,style);
    cleanup = onCleanup(@()close(fig));
    ax = axes(fig); configure_axes(ax,style,style.font.figure4,false); hold(ax,'on');
    [steps,objective,objective_active] = collect_method_trajectory( ...
        e,'proposed','objective');
    [~,feasibility,feasibility_active] = collect_method_trajectory( ...
        e,'proposed','feasibility');
    plot_individuals(ax,steps,objective,style.colors.individual_proposed,style);
    plot_individuals(ax,steps,feasibility,style.colors.individual_feasibility,style);
    plot(ax,steps,active_mean(objective,objective_active), ...
        'Color',style.colors.proposed,'LineWidth',style.lines.mean_width);
    plot(ax,steps,active_mean(feasibility,feasibility_active), ...
        'Color',style.colors.feasibility,'LineWidth',style.lines.mean_width);
    xlabel(ax,'Iteration $k$','Interpreter','latex');
    ylabel(ax,'Value'); xlim(ax,[steps(1) steps(end)]);
    export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
    clear cleanup

    row = contract.IncludeFilename==right_names(case_index);
    fig = make_figure(style.canvas.figure4,style);
    cleanup = onCleanup(@()close(fig));
    ax = axes(fig); configure_axes(ax,style,style.font.figure4,true); hold(ax,'on');
    [steps,proposed,~] = collect_method_trajectory(e,'proposed','distance');
    plot_individuals(ax,steps,proposed,style.colors.individual_proposed,style);
    for m=1:numel(methods)
        [method_steps,values,active] = collect_method_trajectory( ...
            e,methods{m},'distance');
        assert(isequal(method_steps,steps),'Figure 4 step-grid mismatch.');
        plot(ax,steps,active_mean(values,active), ...
            'Color',colors(m,:),'LineWidth',style.lines.mean_width);
    end
    xlabel(ax,'Iteration $k$','Interpreter','latex');
    ylabel(ax,'$D(v_\alpha,v^*)$','Interpreter','latex');
    xlim(ax,[steps(1) steps(end)]);
    export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
    clear cleanup
end
end

function render_figure5(input_files,contract,style)
output_names = {
    ["fig5_0.eps","fig5_5.eps","fig5_50.eps","fig5_100.eps"]
    ["eq1_ellipse_0.eps","eq1_ellipse_5.eps", ...
        "eq1_ellipse_50.eps","eq1_ellipse_100.eps"]
    ["eq1_plane_0.eps","eq1_plane_5.eps", ...
        "eq1_plane_50.eps","eq1_plane_100.eps"]};
expected_ids = ["figure4_5_segment","figure4_5_ellipse","figure4_5_line"];
for case_index=1:3
    e = load_experiment(input_files{case_index});
    assert(string(e.id)==expected_ids(case_index),'Figure 5 file/case mismatch.');
    run = e.results{1,1};
    assert(isequal(run.trajectory.snapshot_steps,[0 5 50 100]), ...
        'Figure 5 snapshot indices are not 0,5,50,100.');
    assert(numel(run.trajectory.snapshots)==4,'Figure 5 snapshot count mismatch.');
    if case_index==1
        a = e.problem_metadata.endpoint_a(:); b = e.problem_metadata.endpoint_b(:);
        assert(isequal(a,[1.6;0.2;0.4]) && isequal(b,[-0.3;-0.7;0.5]), ...
            'Figure 5 segment endpoints differ from the manuscript geometry.');
        u=b-a; t=max(0,min(1,-a.'*u/(u.'*u)));
        assert(norm(a+t*u-e.vstar(:),Inf)<1e-13, ...
            'Figure 5 segment constrained minimizer is inconsistent.');
    end
    for snapshot_index=1:4
        V = run.trajectory.snapshots{snapshot_index};
        expected_dimension = 4-case_index; % segment=3, ellipse/line=2
        if case_index==3, expected_dimension=2; end
        assert(isequal(size(V),[expected_dimension 50]), ...
            'Figure 5 snapshot has wrong dimension or particle count.');
        step = run.trajectory.snapshot_steps(snapshot_index);
        trajectory_index = find(run.trajectory.steps==step,1);
        assert(~isempty(trajectory_index),'Figure 5 consensus step is absent.');
        consensus = run.trajectory.consensus(:,trajectory_index);
        output_name = output_names{case_index}(snapshot_index);
        row = contract.IncludeFilename==output_name;
        if case_index==1
            fig = make_snapshot_figure(style.canvas.figure5_segment,style);
            cleanup = onCleanup(@()close(fig));
            ax = axes(fig); configure_snapshot_axes(ax,style,true); hold(ax,'on');
            plot3(ax,[a(1) b(1)],[a(2) b(2)],[a(3) b(3)],'k-', ...
                'LineWidth',style.lines.geometry_width);
            plot3(ax,V(1,:),V(2,:),V(3,:),'*','Color',style.colors.proposed, ...
                'MarkerSize',style.snapshot.particle_size,'LineWidth',0.65);
            plot3(ax,consensus(1),consensus(2),consensus(3),'*','Color',[1 0 0], ...
                'MarkerSize',style.snapshot.semantic_size, ...
                'LineWidth',style.lines.marker_width);
            plot3(ax,e.vstar(1),e.vstar(2),e.vstar(3),'*','Color',[0 0.6 0], ...
                'MarkerSize',style.snapshot.semantic_size, ...
                'LineWidth',style.lines.marker_width);
            plot3(ax,0,0,0,'k*','MarkerSize',style.snapshot.semantic_size, ...
                'LineWidth',style.lines.marker_width);
            xlim(ax,[-4 4]); ylim(ax,[-4 4]); zlim(ax,[-4 4]);
            view(ax,style.snapshot.view(1),style.snapshot.view(2));
        else
            fig = make_snapshot_figure(style.canvas.figure5_small,style);
            cleanup = onCleanup(@()close(fig));
            ax = axes(fig); configure_snapshot_axes(ax,style,false); hold(ax,'on');
            if case_index==2
                theta=linspace(0,2*pi,400);
                plot(ax,-1+sqrt(2)*cos(theta),sin(theta),'k-', ...
                    'LineWidth',style.lines.geometry_width);
                limits=[-3 3];
            else
                x=linspace(-1,4,300);
                plot(ax,x,3-x,'k-','LineWidth',style.lines.geometry_width);
                limits=[-4 4];
            end
            plot(ax,V(1,:),V(2,:),'*','Color',style.colors.proposed, ...
                'MarkerSize',style.snapshot.small_particle_size,'LineWidth',0.65);
            plot(ax,consensus(1),consensus(2),'*','Color',[1 0 0], ...
                'MarkerSize',style.snapshot.small_semantic_size, ...
                'LineWidth',style.lines.marker_width);
            plot(ax,e.vstar(1),e.vstar(2),'*','Color',[0 0.6 0], ...
                'MarkerSize',style.snapshot.small_semantic_size, ...
                'LineWidth',style.lines.marker_width);
            plot(ax,0,0,'k*','MarkerSize',style.snapshot.small_semantic_size, ...
                'LineWidth',style.lines.marker_width);
            axis(ax,'equal'); xlim(ax,limits); ylim(ax,limits);
        end
        export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
        clear cleanup
    end
end
end

function render_figure6(input_files,configs,contract,style)
output_names = ["line_seg.eps","eq2_dim3_er_1.eps", ...
    "eq2_dim3_er_2.eps","eq2_dim3_er_3.eps","ball.eps", ...
    "eq2_dim20_er_1.eps","eq2_dim20_er_2.eps"];
expected_ids = ["figure6a_d3_case1","figure6b_d3_case3", ...
    "figure6c_d3_case4","figure6d_d3_case5","figure6e_d20_case2", ...
    "figure6f_d20_case3","figure6g_d20_case4"];
methods = {'proposed','quadratic-penalty-cbo','cb2o'};
colors = [style.colors.proposed;style.colors.penalized;style.colors.cb2o];
display_xmax = [500 500 500 500 500 1000 5000];
display_xticks = {[0 250 500],[0 250 500],[0 250 500], ...
    [0 250 500],[0 250 500],[0 500 1000],0:1000:5000};
for case_index=1:7
    e = load_experiment(input_files{case_index});
    assert(string(e.id)==expected_ids(case_index),'Figure 6 file/case mismatch.');
    config = configs(case_index);
    assert(string(config.id)==expected_ids(case_index), ...
        'Figure 6 configuration/case mismatch.');
    assert(isequal(e.methods,methods),'Unexpected Figure 6 method order.');
    assert(isequal(config.methods,methods), ...
        'Unexpected Figure 6 configuration method order.');
    vstar = config.problem.vstar(:);
    row = contract.IncludeFilename==output_names(case_index);
    fig = make_figure(style.canvas.figure6,style);
    cleanup = onCleanup(@()close(fig));
    ax = axes(fig); configure_axes(ax,style,style.font.figure6,true); hold(ax,'on');
    [steps,proposed,~] = collect_consensus_distance_trajectory( ...
        e,'proposed',vstar);
    assert(size(proposed,2)==100,'Figure 6 must contain 100 Proposed trajectories.');
    plot_individuals(ax,steps,proposed,style.colors.individual_proposed,style);
    plotted_positive = proposed(isfinite(proposed) & proposed>0);
    for m=1:3
        [method_steps,values,active] = ...
            collect_consensus_distance_trajectory(e,methods{m},vstar);
        assert(isequal(method_steps,steps),'Figure 6 step-grid mismatch.');
        mean_curve = active_mean(values,active);
        plot(ax,steps,mean_curve, ...
            'Color',colors(m,:),'LineWidth',style.lines.mean_width);
        plotted_positive = [plotted_positive; ...
            mean_curve(isfinite(mean_curve) & mean_curve>0)]; %#ok<AGROW>
    end
    xlabel(ax,'Iteration $k$','Interpreter','latex');
    ylabel(ax,'$D(v_\alpha,v^*)$','Interpreter','latex');
    assert(steps(1)==0 && steps(end)>=display_xmax(case_index), ...
        'Figure 6 stored horizon does not cover the requested display range.');
    xlim(ax,[0 display_xmax(case_index)]);
    xticks(ax,display_xticks{case_index});
    xtickangle(ax,0);
    [data_range,display_range,major_ticks] = set_decade_log_axis( ...
        ax,plotted_positive);
    fprintf(['FIGURE6_YRANGE panel=%c ymin_data=%.16g ymax_data=%.16g ' ...
        'ylim=[%.16g %.16g] yticks=%s\n'],char('a'+case_index-1), ...
        data_range(1),data_range(2),display_range(1),display_range(2), ...
        mat2str(major_ticks,16));
    export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
    clear cleanup
end
end

function render_figure7(input_files,k_values,contract,style)
for case_index=1:numel(k_values)
    e = load_experiment(input_files{case_index});
    assert(string(e.id)==sprintf("figure7_thomson_k%d",k_values(case_index)), ...
        'Figure 7 file/k mismatch.');
    assert(isequal(e.methods,{'proposed-independent-noise'}), ...
        'Unexpected Figure 7 method set.');
    [steps,objectives,active] = collect_method_trajectory( ...
        e,e.methods{1},'objective');
    assert(size(objectives,2)==100,'Figure 7 must contain 100 trajectories.');
    relative_error = abs(objectives-e.objective_star)/abs(e.objective_star);
    relative_error(~active)=NaN;
    mean_curve = active_mean(relative_error,active);
    active_rows = any(active,2);
    assert(any(active_rows),'Figure 7 has no active trajectory samples.');
    last_active_iteration = steps(find(active_rows,1,'last'));
    xmax = min(steps(end),ceil(last_active_iteration/100)*100);
    fprintf('FIGURE7_RANGE k=%d last_active=%g xmax=%g\n', ...
        k_values(case_index),last_active_iteration,xmax);
    output_name = sprintf("eq3_er_%d.eps",k_values(case_index));
    row = contract.IncludeFilename==output_name;
    fig = make_figure(style.canvas.figure7,style);
    cleanup = onCleanup(@()close(fig));
    ax = axes(fig); configure_axes(ax,style,style.font.figure7,true); hold(ax,'on');
    plot_individuals(ax,steps,relative_error,style.colors.individual_proposed,style);
    plot(ax,steps,mean_curve,'Color',style.colors.proposed, ...
        'LineWidth',style.lines.mean_width);
    xlabel(ax,'Iteration $k$','Interpreter','latex');
    ylabel(ax,{'Relative objective','error'});
    xlim(ax,[steps(1) xmax]);
    export_pair(fig,contract.EPSPath(row),contract.PNGPath(row),style);
    clear cleanup
end
end

function experiment = load_experiment(input_file)
assert(isfile(input_file),'Missing authoritative raw file: %s',input_file);
s = load(input_file,'experiment'); experiment = s.experiment;
assert(experiment.is_complete,'Raw experiment is not marked complete: %s',input_file);
assert(experiment.completed_repetitions==experiment.repetitions, ...
    'Raw experiment is incomplete: %s',input_file);
end

function fig = make_figure(canvas,style)
width=canvas(1); height=canvas(2);
fig = figure('Visible','off','Color','w','Units','inches', ...
    'Position',[1 1 width height],'Renderer','painters');
end

function fig = make_snapshot_figure(canvas,style)
fig = make_figure(canvas,style);
end

function configure_axes(ax,style,family_font,use_log)
set(ax,'Color','w','XColor','k','YColor','k','ZColor','k', ...
    'FontName',style.font.name,'FontSize',family_font.tick_size, ...
    'LineWidth',style.axes.line_width,'Box','on','TickDir','out', ...
    'GridColor',style.axes.grid_color,'GridAlpha',style.axes.grid_alpha, ...
    'XGrid','on','YGrid','on','XMinorGrid','off','YMinorGrid','off');
if use_log, ax.YScale='log'; end
ax.XLabel.FontSize=family_font.label_size;
ax.YLabel.FontSize=family_font.label_size;
end

function configure_snapshot_axes(ax,style,is_3d)
set(ax,'Color','w','XColor','k','YColor','k','ZColor','k', ...
    'FontName',style.font.name,'FontSize',style.font.snapshot_tick_size, ...
    'LineWidth',style.axes.line_width,'Box','on','TickDir','out');
if is_3d
    grid(ax,'on'); ax.GridLineStyle='--';
    ax.GridColor=[0.35 0.35 0.35]; ax.GridAlpha=style.snapshot.segment_grid_alpha;
else
    grid(ax,'off');
end
end

function plot_individuals(ax,steps,values,color,style)
assert(size(values,2)==100,'Expected exactly 100 individual trajectories.');
for r=1:size(values,2)
    plot(ax,steps,values(:,r),'Color',color, ...
        'LineWidth',style.lines.individual_width,'HandleVisibility','off');
end
end

function curve = active_mean(values,active)
assert(isequal(size(values),size(active)),'Active-mask size mismatch.');
masked = values;
masked(~active) = NaN;
curve = mean(masked,2,'omitnan');
curve(sum(active,2)==0) = NaN;
end

function [data_range,display_range,major_ticks] = set_decade_log_axis(ax,values)
values = values(isfinite(values) & values>0);
assert(~isempty(values),'Logarithmic panel has no positive finite data.');
data_range = [min(values) max(values)];
lo_exp = floor(log10(data_range(1)));
hi_exp = ceil(log10(data_range(2)));
if lo_exp==hi_exp, lo_exp=hi_exp-1; end
display_range = 10.^[lo_exp hi_exp];
assert(display_range(1)<=data_range(1) && ...
    display_range(2)>=data_range(2),'Rounded log range clips plotted data.');
decade_count = hi_exp-lo_exp+1;
if decade_count<=7
    tick_exponents = lo_exp:hi_exp;
else
    stride = ceil((hi_exp-lo_exp)/6);
    tick_exponents = unique([lo_exp:stride:hi_exp hi_exp]);
end
major_ticks = 10.^tick_exponents;
ylim(ax,display_range);
yticks(ax,major_ticks);
end

function export_pair(fig,eps_path,png_path,style)
eps_path = char(eps_path); png_path = char(png_path);
fig.Renderer='painters';
exportgraphics(fig,eps_path,'ContentType','vector','BackgroundColor','white');
exportgraphics(fig,png_path,'Resolution',style.preview_resolution, ...
    'BackgroundColor','white');
end
